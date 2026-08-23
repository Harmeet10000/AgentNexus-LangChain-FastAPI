"""Retry policy for the KB LangGraph input/output boundaries.

This module owns exactly one decision: *which* failures a KB boundary call is allowed to
try again, *how long* it waits between tries, and *what type* the caller sees once the
budget is spent. Everything else — model retries, tool retries, replay safety — is
deliberately somebody else's layer (model/tool retries belong to middleware; replay
safety belongs to idempotency keys, never to attempt counters, because a node-local
counter resets on checkpoint replay and silently multiplies the budget).

Three defects are corrected here, and each one shipped for a reason worth recording.

**1. The retry predicate was keyed on the base exception type.** Every failure was
retryable, which is a different statement from "failures are retryable": a malformed
argument, a schema violation, a permission denial and a bug in our own code were all
retried, three times, and then relabelled. Retrying a deterministic failure cannot
succeed; it only triples the latency of the error report and triples the load on a
dependency that already said no. The retryable set is now *named* — see
``TRANSIENT_EXTERNAL_TYPES`` and ``_RETRYABLE_STATUS_CODES`` — and naming it is the
point: a set you cannot enumerate is a set you cannot reason about.

**2. The wait was zero.** Three attempts fired back-to-back in roughly no wall-clock
time at all. Against a rate-limited endpoint that produces three refusals instead of
one, so the "retry" was strictly worse than no retry: same outcome, triple the quota
burn, and a shape that reads in the logs like resilience. Growing wait with jitter is
the library's own idiom and the only one that lets a dependency actually recover.

**3. The catch-all swallowed the graph framework's control-flow pause.** This is the
subtle one and it is why the predicate could not simply be left broad. LangGraph does
not pause by returning a sentinel — it pauses **by raising**, and the whole pause family
(the subgraph interrupt, the deprecated node-raised interrupt, and the parent-command
bubble) descends from a single base class that itself descends from ``Exception``. A
predicate keyed on ``Exception`` therefore treats a pause as a failure: it retries the
node body two more times — re-running whatever side effects already landed — and then
relabels the pause as an external failure, at which point the graph never pauses and the
resume value is lost. Ingestion raises no pause *today*; change 3 introduces one, and a
pause that is silently eaten presents as "interrupts don't work" with no traceback
pointing here. ``GraphBubbleUp`` is excluded from the retryable set *and* from the
wrapping, so it leaves this function as the exact object that entered it.

**The remedy that was tried first and could not work, recorded so it is not re-proposed.**
An earlier contract said the boundary should chain the original failure via ``raise …
from exc`` "so a caller's existing degradation branch still matches". It does not.
Chaining populates ``__cause__``; it does **not** change the type of the object raised.
The callers in this package degrade on the language-model framework's base exception, and
no amount of chaining makes ``TransientExternalError`` an instance of that base — so the
degradation branches could never fire for a wrapped call, and the pipeline propagated
exactly where it appeared to degrade. Chaining and type-preservation are two different
properties and only one of them was being delivered. The chosen contract keeps the single
named transient type at the boundary and **converts the callers** to catch it alongside
the framework exception they already caught. That makes the caller list greppable instead
of assumed, which matters because a caller that is missed is a degradation branch that
silently stops firing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from langchain_core.exceptions import OutputParserException
from langgraph.errors import GraphBubbleUp
from redis import exceptions as redis_exceptions
from sqlalchemy import exc as sqlalchemy_exc
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Final

    from tenacity import RetryCallState


class TransientExternalError(Exception):
    """The one type a KB boundary raises when a retryable failure outlived its budget.

    Deliberately a single type rather than a family. The original failure is always
    reachable through ``__cause__`` — with its own type, message and notes intact — so
    callers that need to tell a quota refusal from a malformed response from a dependency
    that went away read the cause rather than asking this class to carry a taxonomy it
    would immediately fall behind.
    """


#: Failure types that are transient *by construction* — the dependency is reachable-in-
#: principle and the same call may succeed unchanged. Kept deliberately narrow and
#: transport-shaped rather than provider-shaped: naming provider SDK exception classes
#: here would couple this shared boundary to whichever vendor a node happens to call, and
#: the transport layer underneath every one of those SDKs already raises these.
#:
#: ``OutputParserException`` earns its place for a different reason than the rest: a
#: structured-response parse failure is not a network event, but the generation is
#: nondeterministic, so re-asking is a genuine remedy rather than a hopeful repeat. It is
#: named specifically and not by way of its framework base class, because that base also
#: covers deterministic configuration errors that retrying cannot fix.
TRANSIENT_EXTERNAL_TYPES: Final[tuple[type[BaseException], ...]] = (
    TimeoutError,  # also covers asyncio.TimeoutError, an alias since 3.11
    ConnectionError,
    httpx.TransportError,  # connect/read/write/pool timeouts and network errors
    redis_exceptions.ConnectionError,
    redis_exceptions.TimeoutError,
    sqlalchemy_exc.OperationalError,  # connection dropped, server shutting down
    sqlalchemy_exc.InterfaceError,  # connection invalidated below the DBAPI
    OutputParserException,
)

#: Statuses that mean "ask again later", checked structurally so that any dependency
#: reporting a status — an HTTP client, a cloud SDK that maps gRPC codes onto HTTP
#: statuses, or this project's own exception hierarchy — is covered without being
#: imported. Authentication and authorisation refusals (401/403) are pointedly absent:
#: credentials do not become valid by waiting, so retrying them only delays a report that
#: needs a human. 404 and 422 are absent for the same reason.
_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})

#: First wait, in seconds. Doubles each attempt.
_INITIAL_WAIT_SECONDS: Final[float] = 0.5

#: Ceiling on a single wait, so a spent budget still reports in bounded time.
_MAX_WAIT_SECONDS: Final[float] = 8.0

#: Jitter spread, in seconds. Held **strictly below** the initial wait on purpose: the
#: smallest gap between consecutive waits equals the initial wait, so a spread smaller
#: than that keeps the sequence strictly increasing for every possible draw while still
#: de-correlating concurrent callers. A spread equal to or larger than the initial wait
#: would let a later attempt legitimately wait less than an earlier one, which is a
#: property that is hard to distinguish from the no-wait defect this replaces.
_JITTER_SECONDS: Final[float] = 0.25


def describe_failure(exc: BaseException) -> str:
    """Render a failure so a degradation record names the *original* problem.

    A degraded branch that logs only the exception it caught becomes uninformative the
    moment the boundary wraps: the transient type's message names the operation and the
    attempt count and says nothing about *why* the dependency refused. Walking
    ``__cause__`` puts the original type and message back into the record, which is the
    entire reason the boundary chains rather than discarding.

    The traversal carries a seen-set because ``__cause__`` is caller-assignable and a
    cycle would otherwise hang the logging path — a degradation handler is the last place
    that can afford to be the thing that fails.
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    seen = {id(exc)}
    cause = exc.__cause__
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        parts.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " <- caused by ".join(parts)


def _status_code_of(exc: BaseException) -> int | None:
    """Recover an integer status from an exception without knowing its library.

    Three shapes cover everything this pipeline calls: a status attribute on the
    exception (this project's own hierarchy, and most HTTP SDKs), a numeric ``code``
    (cloud SDKs that map their transport codes onto HTTP statuses), and a status carried
    on an attached response object (the HTTP client used underneath the SDKs).

    Only ``int`` values are accepted. Several libraries use ``code`` for a *string*
    identifier, and a string compared against a set of integers would silently never
    match — an outcome indistinguishable from "not retryable", which is precisely the
    kind of quiet wrong answer this module exists to stop producing.
    """
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def is_transient(exc: BaseException) -> bool:
    """Decide whether ``exc`` is worth another attempt.

    The pause check comes first and is not merely an optimisation: the graph framework's
    control-flow pause descends from ``Exception``, so any later test that widens even
    slightly would start capturing it. Ordering the exclusion ahead of every positive
    test makes that impossible to reintroduce by accident.
    """
    if isinstance(exc, GraphBubbleUp):
        return False
    if isinstance(exc, TransientExternalError):
        # Already labelled by an inner boundary. Re-retrying a spent budget would
        # multiply the attempt count by the nesting depth.
        return False
    if isinstance(exc, TRANSIENT_EXTERNAL_TYPES):
        return True
    return _status_code_of(exc) in _RETRYABLE_STATUS_CODES


def _log_before_sleep(label: str, attempts: int) -> Callable[[RetryCallState], None]:
    """Build the between-attempts log hook.

    ``outcome`` is optional on the retry state, so it is checked rather than asserted:
    the hook only ever runs after a failed attempt, but a type-checker cannot know that
    and suppressing the complaint would leave a comment that rots into a lie the moment
    the library changes shape.
    """

    def _hook(state: RetryCallState) -> None:
        outcome = state.outcome
        error = outcome.exception() if outcome is not None else None
        logger.bind(
            label=label,
            attempt=state.attempt_number,
            attempts=attempts,
            wait_seconds=state.idle_for,
        ).warning("kb_retry_transient_retry", error=str(error))

    return _hook


async def retry_immediate[T](
    operation: Callable[[], Awaitable[T]],
    *,
    label: str,
    attempts: int = 3,
) -> T:
    """Run one input/output boundary call under the named-transient retry policy.

    Wraps a **client call**, never a graph node body. That restriction is what keeps the
    control-flow pause exclusion below sufficient: a node body may pause, and while the
    exclusion means a pause raised inside here still escapes intact, a retry wrapper
    around a whole node would re-run the node's completed side effects on every attempt.

    Failures divide three ways:

    * a transient failure that later succeeds — returns normally, no trace in the result;
    * a transient failure that outlives ``attempts`` — becomes ``TransientExternalError``
      with the final original failure as its ``__cause__``;
    * anything else, including the framework's control-flow pause — propagates on the
      **first** attempt, unretried and untouched, with its own type and traceback.

    The name is now a slight historical misnomer: the waits are no longer immediate. It
    is kept because renaming it would rewrite every call site in the pipeline for no
    behavioural gain, and the docstring is a better place to record the change than a
    churned import list.
    """
    retryer = AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(
            initial=_INITIAL_WAIT_SECONDS,
            max=_MAX_WAIT_SECONDS,
            jitter=_JITTER_SECONDS,
        ),
        retry=retry_if_exception(is_transient),
        reraise=True,
        before_sleep=_log_before_sleep(label, attempts),
    )
    try:
        async for attempt in retryer:
            with attempt:
                return await operation()
    except Exception as exc:
        if not is_transient(exc):
            # Not ours to relabel. A control-flow pause reaches here on its first
            # attempt and must leave as the same object: re-typing it is what makes a
            # graph silently refuse to pause. A deterministic failure reaches here for
            # the same reason and keeps its own type so the caller can tell a bad
            # request from a dependency outage.
            raise
        msg = f"{label} failed after {attempts} attempts"
        raise TransientExternalError(msg) from exc
    # Unreachable in practice: the loop either returns or the retryer raises. Kept as a
    # raise rather than a fall-through so a future library change that ends the iteration
    # without an outcome fails loudly instead of returning ``None`` as a ``T``.
    msg = f"{label} ended without a result or an exception"
    raise TransientExternalError(msg)
