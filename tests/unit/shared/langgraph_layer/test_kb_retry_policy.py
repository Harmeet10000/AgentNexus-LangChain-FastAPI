"""Unit tests for task C5 — the KB retry policy names its retryable set and waits.

`retry_immediate` is the highest-fan-in untested function in this change: every
input/output call in the pipeline being promoted goes through it, so a wrong policy here
is a wrong policy in eleven places at once. C5's test is marked Mandatory for that reason.

Three separate claims are pinned, one per defect the task corrects:

* **(a)** a failure *outside* the retryable set propagates on the first attempt. The old
  predicate was keyed on the base exception type, so a deterministic failure — a bad
  argument, a schema violation, a bug of ours — was tried three times and then relabelled.
* **(b)** a *named* transient failure retries to the configured count with a **growing**
  wait. The old wait strategy was the no-wait one, so three attempts against a
  rate-limited endpoint produced three refusals in roughly zero milliseconds: the same
  outcome as no retry, at triple the quota cost, wearing the shape of resilience.
* **(c)** the graph framework's control-flow pause propagates **immediately and
  unchanged**. LangGraph pauses by *raising*, and the whole pause family descends from
  `GraphBubbleUp`, which descends from `Exception` — so the old catch-all treated a pause
  as a failure, retried the surrounding call, and then relabelled the pause as an external
  error. Ingestion raises no pause today; change 3 adds one, and the symptom then is
  "interrupts don't work" with nothing in the traceback pointing at the retry wrapper.

**No test here measures elapsed time.** A duration threshold measures the test runner, not
the policy: it is flaky under load and it passes for the wrong reason on a fast machine.
The waits are proven by *intercepting* the retry loop's sleep and asserting the sequence
of durations it was asked for, and the attempt behaviour is proven by counting invocations
of the wrapped operation. Both are exact rather than approximate.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from langchain_core.exceptions import LangChainException, OutputParserException
from langgraph.errors import GraphBubbleUp, GraphInterrupt, ParentCommand
from langgraph.types import Command

from app.shared.langgraph_layer import kb_retry
from app.shared.langgraph_layer.kb_retry import TransientExternalError, retry_immediate

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

_READ_TIMED_OUT = "read timed out"


@contextlib.contextmanager
def _intercepted_sleep() -> Iterator[list[float]]:
    """Run the retry loop with its sleep replaced by a recorder.

    The loop calls `self.sleep(...)` on the retrying object, and that attribute is set in
    the constructor — so the seam is the *class* the module reaches for, not a module-level
    sleep function. Patching the class in `kb_retry`'s own namespace with a subclass that
    swaps its sleep leaves the real wait strategy, stop condition and retry predicate
    untouched, which is the point: the durations recorded are the ones production would
    actually have slept for, not durations this test chose.
    """
    recorded: list[float] = []

    async def _record(seconds: float) -> None:
        recorded.append(float(seconds))

    class _Recording(kb_retry.AsyncRetrying):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.sleep = _record

    with patch.object(kb_retry, "AsyncRetrying", _Recording):
        yield recorded


def _counting_failure(error: BaseException) -> tuple[list[int], Any]:
    """An operation that always fails with `error`, and a list recording each entry."""
    calls: list[int] = []

    async def _operation() -> object:
        calls.append(len(calls) + 1)
        raise error

    return calls, _operation


# --- (a) A failure outside the retryable set propagates on the first attempt ---


async def test_non_retryable_failure_is_attempted_exactly_once() -> None:
    calls, operation = _counting_failure(ValueError("malformed page range"))

    with _intercepted_sleep() as slept, pytest.raises(ValueError, match="malformed page range"):
        await retry_immediate(operation, label="parse_document", attempts=3)

    assert calls == [1], "a deterministic failure must not be tried again"
    assert slept == [], "a single attempt has nothing to wait between"


async def test_non_retryable_failure_keeps_its_own_type_and_identity() -> None:
    """The caller must be able to tell a bad request from a dependency outage.

    Relabelling every failure as transient destroyed exactly that distinction, and it did
    so silently — the wrapper's message names the operation and the attempt count, which
    reads like a diagnosis.
    """
    original = ValueError("chunk index out of range")
    _, operation = _counting_failure(original)

    with _intercepted_sleep(), pytest.raises(ValueError) as exc_info:
        await retry_immediate(operation, label="store_clause", attempts=3)

    assert exc_info.value is original
    assert not isinstance(exc_info.value, TransientExternalError)


async def test_an_authentication_refusal_is_not_retried() -> None:
    """Credentials do not become valid by waiting, so 401 is outside the retryable set."""

    class _UnauthorizedError(Exception):
        status_code = 401

    calls, operation = _counting_failure(_UnauthorizedError("invalid api key"))

    with _intercepted_sleep() as slept, pytest.raises(_UnauthorizedError):
        await retry_immediate(operation, label="gemini_segment_document", attempts=3)

    assert calls == [1]
    assert slept == []


async def test_the_framework_base_exception_alone_is_not_retryable() -> None:
    """Narrowing to the model framework's base class would have re-created the defect.

    That base also covers deterministic configuration errors, so treating it as transient
    is the same mistake one level down. Only the parse refusal below it is retryable, and
    it is named individually for that reason.
    """
    assert not kb_retry.is_transient(LangChainException("model not configured"))
    assert kb_retry.is_transient(OutputParserException("expected JSON, got prose"))


# --- (b) A named transient failure retries to the configured count with a growing wait ---


async def test_named_transient_failure_retries_to_the_configured_count() -> None:
    calls, operation = _counting_failure(TimeoutError("read timed out"))

    with _intercepted_sleep(), pytest.raises(TransientExternalError):
        await retry_immediate(operation, label="embed_texts", attempts=3)

    assert calls == [1, 2, 3]


async def test_the_wait_between_attempts_grows_and_is_never_zero() -> None:
    """The direct refutation of the no-wait defect, asserted on the requested durations.

    Four attempts rather than three, so monotonicity is asserted across three gaps instead
    of one — a single comparison can pass by coincidence in a way a chain of them cannot.
    """
    calls, operation = _counting_failure(ConnectionError("connection reset by peer"))

    with _intercepted_sleep() as slept, pytest.raises(TransientExternalError):
        await retry_immediate(operation, label="postgres_store_clause", attempts=4)

    assert calls == [1, 2, 3, 4]
    assert len(slept) == 3, "four attempts leave exactly three gaps to wait in"
    assert all(wait > 0 for wait in slept), "a zero wait is the defect being removed"
    assert slept[0] < slept[1] < slept[2], f"waits must grow, got {slept}"


async def test_the_growing_wait_is_bounded() -> None:
    """Growth without a ceiling turns a spent budget into an unbounded stall."""
    _, operation = _counting_failure(TimeoutError("read timed out"))

    with _intercepted_sleep() as slept, pytest.raises(TransientExternalError):
        await retry_immediate(operation, label="embed_texts", attempts=8)

    assert all(wait <= kb_retry._MAX_WAIT_SECONDS for wait in slept), f"unbounded: {slept}"


async def test_a_quota_refusal_is_retried_without_naming_a_vendor_exception() -> None:
    """Quota refusals arrive as a vendor type, so the set is closed structurally too.

    Naming provider SDK classes in the shared boundary would couple it to whichever vendor
    a node happens to call; every one of them reports a status, so the status is what is
    matched.
    """

    class _ResourceExhaustedError(Exception):
        code = 429

    calls, operation = _counting_failure(_ResourceExhaustedError("quota exceeded"))

    with _intercepted_sleep() as slept, pytest.raises(TransientExternalError):
        await retry_immediate(operation, label="gemini_contextualize_chunk", attempts=3)

    assert calls == [1, 2, 3]
    assert slept[0] < slept[1]


async def test_a_transient_failure_that_recovers_returns_without_a_trace() -> None:
    attempts_seen: list[int] = []

    async def _operation() -> str:
        attempts_seen.append(len(attempts_seen) + 1)
        if len(attempts_seen) < 3:
            raise TimeoutError(_READ_TIMED_OUT)
        return "parsed"

    with _intercepted_sleep() as slept:
        result = await retry_immediate(_operation, label="docling_parse", attempts=3)

    assert result == "parsed"
    assert attempts_seen == [1, 2, 3]
    assert len(slept) == 2


# --- (c) The control-flow pause propagates immediately and unchanged ---


async def test_the_pause_hierarchy_hangs_off_a_subclass_of_exception() -> None:
    """The premise of (c), asserted rather than assumed.

    If this ever stops holding, the exclusion below becomes dead code that still reads as
    a guard — so the reason the guard is *needed* is pinned alongside the guard itself.
    """
    assert issubclass(GraphBubbleUp, Exception)
    assert issubclass(GraphInterrupt, GraphBubbleUp)
    assert issubclass(ParentCommand, GraphBubbleUp)
    assert not kb_retry.is_transient(GraphInterrupt())


async def test_a_subgraph_pause_propagates_on_the_first_attempt() -> None:
    pause = GraphInterrupt()
    calls, operation = _counting_failure(pause)

    with _intercepted_sleep() as slept, pytest.raises(GraphInterrupt) as exc_info:
        await retry_immediate(operation, label="gemini_contextualize_chunk", attempts=3)

    assert exc_info.value is pause, "the pause must arrive as the same object it left as"
    assert calls == [1], "retrying a pause re-runs side effects that already landed"
    assert slept == []
    assert not isinstance(exc_info.value, TransientExternalError)


async def test_a_parent_command_pause_propagates_on_the_first_attempt() -> None:
    """The other member of the family, so the exclusion is on the base and not one leaf."""
    pause = ParentCommand(Command(goto="finalize"))
    calls, operation = _counting_failure(pause)

    with _intercepted_sleep() as slept, pytest.raises(ParentCommand) as exc_info:
        await retry_immediate(operation, label="graphiti_add_episode", attempts=3)

    assert exc_info.value is pause
    assert calls == [1]
    assert slept == []


async def test_a_pause_raised_after_a_transient_failure_still_escapes_intact() -> None:
    """The interleaved case, which the single-failure tests cannot reach.

    Once a retry is already in flight the loop is mid-policy, and a pause arriving then is
    the shape most likely to be mistaken for "the retry failed again".
    """
    pause = GraphInterrupt()
    seen: list[str] = []

    async def _operation() -> object:
        if not seen:
            seen.append("transient")
            raise TimeoutError(_READ_TIMED_OUT)
        seen.append("pause")
        raise pause

    with _intercepted_sleep() as slept, pytest.raises(GraphInterrupt) as exc_info:
        await retry_immediate(_operation, label="gemini_entity_extraction", attempts=3)

    assert seen == ["transient", "pause"]
    assert exc_info.value is pause
    assert len(slept) == 1, "one gap for the one transient failure, none for the pause"


# --- The policy does not compound when boundaries nest ---


async def test_an_already_labelled_transient_failure_is_not_retried_again() -> None:
    """Otherwise the effective attempt budget is the product of the nesting depth."""
    calls, operation = _counting_failure(TransientExternalError("inner budget spent"))

    with _intercepted_sleep() as slept, pytest.raises(TransientExternalError):
        await retry_immediate(operation, label="outer", attempts=3)

    assert calls == [1]
    assert slept == []
