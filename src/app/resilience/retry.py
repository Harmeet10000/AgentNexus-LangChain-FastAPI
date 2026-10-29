"""Tenacity helpers for transient-failure retries.

The third member of the resilience family beside ``circuit_breaker`` (what
happens when a call keeps failing) and ``idempotency`` (what happens when a
call replays): this module owns *which* failures may be attempted again, how
long the waits are, and what the caller sees when the budget is spent.

Design rules this module enforces repo-wide:

1. **The retryable set is named.** Retrying a deterministic failure (schema
   violation, permission denial, our own bug) cannot succeed; it only
   multiplies latency and load. ``TRANSIENT_EXCEPTION_TYPES`` is enumerable on
   purpose, and kernel ``FeatureError`` subclasses with ``retryable = True``
   are honored through :func:`is_transient`.
2. **Waits grow with jitter.** Back-to-back attempts against a rate-limited
   dependency produce refusals, not resilience. Every policy here waits
   exponentially with jitter.
3. **Reraise, never wrap.** ``reraise=True`` everywhere: type-preservation is
   the property callers depend on (a degradation branch matches on type;
   ``raise ... from`` changes neither the type nor the caller's match). See
   ``app.ai.graph.retry`` for the full worked example of why wrapping fails.
4. **Result-aware retries stop on the failure's own verdict.** For
   ``returns.result`` callables the kernel ``retryable`` flag decides — a
   non-retryable ``Failure`` fails fast even when the budget is unspent.

Feature-local policies with stricter contracts (e.g. the LangGraph
boundary module ``app.ai.graph.retry``, which additionally excludes
graph control-flow pauses) stay where they are; they may adopt the
predicates here rather than re-deriving them.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
from returns.result import Failure
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry,
    retry_if_exception,
    retry_if_result,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.kernel.errors import FeatureError
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from returns.result import Result
    from tenacity import RetryCallState
    from tenacity.wait import wait_base

__all__ = [
    "DEFAULT_MAX_WAIT_SECONDS",
    "DEFAULT_STOP_AFTER_ATTEMPTS",
    "TRANSIENT_EXCEPTION_TYPES",
    "default_wait",
    "is_retryable_failure",
    "is_transient",
    "retry_result",
    "transient_retry",
]

#: The named transient set. Transport-level failures and timeouts only —
#: deterministic failures (validation, authorization, our own bugs) are never
#: members. Kernel ``FeatureError`` subclasses advertise retryability through
#: their ``retryable`` flag instead of membership here.
TRANSIENT_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    httpx.TransportError,
    httpx.TimeoutException,
)

#: Attempt budget and wait-shape defaults. Overridable per call site; the
#: values are chosen so a transient blip recovers inside ~2s while a hard
#: outage fails in bounded time.
DEFAULT_STOP_AFTER_ATTEMPTS = 3
DEFAULT_MAX_WAIT_SECONDS = 5.0


def is_transient(exc: BaseException | FeatureError) -> bool:
    """Decide whether *exc* may be attempted again.

    A failure is transient when it is a member of
    :data:`TRANSIENT_EXCEPTION_TYPES` or when its type opts in explicitly via
    the kernel ``FeatureError.retryable`` flag. ``FeatureError`` is a Pydantic
    model rather than a ``BaseException``, so the union is the honest input
    type. Control-flow signals that
    descend from ``Exception`` (LangGraph pauses) are NOT members here — a
    boundary that can observe a pause must exclude it explicitly, the way
    ``app.ai.graph.retry`` does.
    """
    if isinstance(exc, TRANSIENT_EXCEPTION_TYPES):
        return True
    return isinstance(exc, FeatureError) and exc.retryable


def is_retryable_failure(failure: BaseException | FeatureError) -> bool:
    """Result-side twin of :func:`is_transient`.

    ``returns.result`` failures carry the error object, so the same verdict
    applies on the Failure branch of a ``Result``-returning callable.
    """
    return is_transient(failure)


def default_wait() -> wait_base:
    """Exponential wait with jitter — the only wait shape that lets a
    dependency actually recover between attempts."""
    return wait_exponential_jitter(initial=0.2, max=DEFAULT_MAX_WAIT_SECONDS)


def _before_sleep_log(retry_state: RetryCallState) -> None:
    """Surface every scheduled retry as a named event, never a success line."""
    outcome = retry_state.outcome
    failure_name = (
        type(outcome.exception()).__name__ if outcome is not None and outcome.failed else "retryable_failure"
    )
    logger.bind(
        attempt=retry_state.attempt_number,
        waited=retry_state.seconds_since_start,
        outcome=failure_name,
    ).warning("retry_scheduled")


def transient_retry(
    *,
    stop: Any | None = None,
    wait: wait_base | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: retry a callable on transient failures, reraising the last.

    The raised exception is the *original* object (``reraise=True``), so
    caller degradation branches keyed on exception type keep firing.
    """
    return retry(
        retry=retry_if_exception(is_transient),
        stop=stop or stop_after_attempt(DEFAULT_STOP_AFTER_ATTEMPTS),
        wait=wait or default_wait(),
        before_sleep=_before_sleep_log,
        reraise=True,
    )


async def retry_result(
    operation: Callable[[], Awaitable[Result[Any, Any]]],
    *,
    stop: Any | None = None,
    wait: wait_base | None = None,
) -> Result[Any, Any]:
    """Retry a ``Result``-returning callable while its Failure is retryable.

    Success returns immediately. A non-retryable ``Failure`` is returned to
    the caller without spending the remaining budget. When the budget is
    exhausted the last ``Failure`` is returned — never raised, never wrapped —
    so the caller's ``match`` branch keeps working unchanged.
    """
    retrying = AsyncRetrying(
        retry=retry_if_result(_is_retryable_outcome),
        stop=stop or stop_after_attempt(DEFAULT_STOP_AFTER_ATTEMPTS),
        wait=wait or default_wait(),
        before_sleep=_before_sleep_log,
        # reraise=True is WRONG here: with a *result* predicate, tenacity's
        # budget-exhaustion path does `raise last_attempt.result()` — and the
        # final result is a Failure *value*, not an exception, so it crashes
        # with TypeError instead of returning. reraise=False plus the explicit
        # branch below preserves rule 3 (original exception type) for the
        # exception case while returning the last Failure for the result case.
        reraise=False,
    )
    try:
        return await retrying(operation)
    except RetryError as exc:
        last = exc.last_attempt
        if last.failed:
            raise last.result() from None  # original exception, type preserved
        return last.result()  # budget spent on retryable Failures: return the last


def _is_retryable_outcome(outcome: Result[Any, Any]) -> bool:
    """Tenacity result predicate: retry while the Failure itself is retryable."""
    return isinstance(outcome, Failure) and is_retryable_failure(outcome.failure())
