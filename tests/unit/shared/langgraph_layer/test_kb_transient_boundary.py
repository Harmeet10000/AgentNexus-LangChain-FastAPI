"""Unit tests for task C6 — one typed transient failure, and callers that actually catch it.

**C6 replaces a remedy that could not work, and these tests exist so the reason cannot be
re-lost.** The earlier contract said the boundary should chain the original failure via
`raise … from exc` "so a caller's existing degradation branch still matches". It does not.
Chaining populates `__cause__`; it does **not** change the type of the object raised. The
degradation branches in `ingestion_kb/nodes.py` matched on the model framework's base
exception, and `TransientExternalError` is not an instance of that base however carefully
the chain is built — so those branches could not fire for any wrapped call. The pipeline
propagated at exactly the three points where reading the code said it degraded.

The chosen contract is the other coherent one: the boundary raises **one** typed transient
failure chained to the original, and **every caller with a degradation branch around a
retried operation is converted** to catch it. That makes the caller list a greppable step
rather than an invisible assumption, which matters because a caller that is missed is a
degradation branch that silently stops firing — a defect with no symptom until the day it
was supposed to save something.

The decisive test here is `test_an_exhausted_retry_reaches_the_converted_degradation_branch`
and its two siblings: they drive a **real** exhausted retry — the real policy, the real
wrapping, the real `except` — through each converted caller. Nothing about the boundary is
mocked except the durations it would have slept for. The earlier contract fails all three.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.exceptions import LangChainException, OutputParserException

from app.shared.langgraph_layer import kb_retry
from app.shared.langgraph_layer.ingestion_kb import nodes as nodes_module
from app.shared.langgraph_layer.ingestion_kb.nodes import (
    make_classify_extract_node,
    make_contextualize_chunk_node,
    make_segment_document_node,
)
from app.shared.langgraph_layer.ingestion_kb.state import (
    ClauseSegment,
    ContractMetadata,
    IngestionState,
    ParsedDocument,
)
from app.shared.langgraph_layer.kb_retry import (
    TransientExternalError,
    describe_failure,
    retry_immediate,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

_DOC_ID = "doc-42"
_CLAUSE_ID = "clause-7"
_CHUNK_INDEX = 3


class _UnauthorizedError(Exception):
    """A vendor authentication refusal: outside the retryable set, so it is never wrapped."""

    status_code = 401


class _QuotaExceededError(Exception):
    """A vendor quota refusal: inside the retryable set by status, so it is wrapped."""

    code = 429


@contextlib.contextmanager
def _no_real_waiting() -> Iterator[list[float]]:
    """Let the real retry policy run without spending its waits.

    The policy, the predicate and the wrapping are all genuine here — only the sleep is
    intercepted. Without this the C6 tests below would spend the boundary's full backoff
    budget on every exhausted retry, which is a slow suite paying for nothing: the
    durations are C5's subject, not C6's.
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


def _always_failing_llm(error: BaseException) -> MagicMock:
    """A structured runnable whose every invocation raises `error`."""
    llm = MagicMock()

    async def _ainvoke(_messages: Any) -> object:
        raise error

    llm.ainvoke = _ainvoke
    return llm


def _segment() -> ClauseSegment:
    return ClauseSegment(
        clause_id=_CLAUSE_ID,
        text="The Supplier shall indemnify the Customer against all third-party claims.",
        chunk_index=_CHUNK_INDEX,
        page_no=2,
    )


def _segmentation_state() -> IngestionState:
    return IngestionState(
        doc_id=_DOC_ID,
        parsed_document=ParsedDocument(
            markdown="1. Indemnity\n\nThe Supplier shall indemnify the Customer.",
            title="Supply Agreement",
            source="upload",
        ),
        contract_metadata=ContractMetadata(),
    )


def _contextualize_payload() -> dict[str, Any]:
    return {
        "doc_id": _DOC_ID,
        "segment": _segment().model_dump(),
        "contract_metadata": {},
        "source": "upload",
    }


# --- Proof 2: the cause carries the original type and message ---


async def test_the_wrapped_failure_chains_the_original_exception_object() -> None:
    original = OutputParserException("expected JSON, got prose")

    async def _operation() -> object:
        raise original

    with _no_real_waiting(), pytest.raises(TransientExternalError) as exc_info:
        await retry_immediate(_operation, label="gemini_segment_document", attempts=2)

    assert exc_info.value.__cause__ is original


async def test_the_original_type_and_message_are_recoverable_from_the_cause() -> None:
    """A single opaque type at the boundary is only acceptable if nothing is lost behind it.

    The wrapper's own message names the operation and the attempt count and says nothing
    about *why* the dependency refused — which reads like a diagnosis and is not one. The
    cause is where the answer lives, so the cause is what is asserted.
    """
    original = OutputParserException("expected JSON, got prose")

    async def _operation() -> object:
        raise original

    with _no_real_waiting(), pytest.raises(TransientExternalError) as exc_info:
        await retry_immediate(_operation, label="gemini_segment_document", attempts=2)

    cause = exc_info.value.__cause__
    assert type(cause) is OutputParserException
    assert "expected JSON, got prose" in str(cause)
    assert "gemini_segment_document" in str(exc_info.value)
    # And the rendering used by the degradation logs reaches through to it, so the record
    # a human reads is not the wrapper's message alone.
    assert "expected JSON, got prose" in describe_failure(exc_info.value)
    assert OutputParserException.__name__ in describe_failure(exc_info.value)


# --- Proof 3: an exhausted retry reaches the converted caller's degradation branch ---
# This is the scenario the old contract failed. All three converted callers are driven,
# because "the boundary was fixed" and "every caller was converted" are separate claims and
# only the second one is at risk of being half-done.


async def test_an_exhausted_retry_reaches_the_converted_degradation_branch() -> None:
    """Segmentation: the branch must run its fallback, not propagate.

    The failure is a genuine transient one, the retry genuinely exhausts, and the
    `TransientExternalError` that arrives at the `except` is the real one the boundary
    built. Under the earlier contract this test raises instead of degrading.
    """
    original = OutputParserException("expected JSON, got prose")
    node = make_segment_document_node(_always_failing_llm(original))
    logger_double = MagicMock()

    with (
        _no_real_waiting(),
        patch.object(nodes_module, "logger", logger_double),
    ):
        result = await node(_segmentation_state())

    # The degradation branch executed: the deterministic fallback produced segments.
    assert result.get("segments"), "the fallback segmentation did not run"

    # And the record names the original failure, reached through the cause.
    logged = logger_double.bind.call_args.kwargs["error"]
    assert "expected JSON, got prose" in logged
    assert OutputParserException.__name__ in logged
    logger_double.bind.return_value.warning.assert_called_once_with(
        "structured_segmentation_failed_using_fallback"
    )


async def test_an_exhausted_retry_reaches_the_contextualize_degradation_branch() -> None:
    original = _QuotaExceededError("quota exceeded for model")
    node = make_contextualize_chunk_node(_always_failing_llm(original))
    logger_double = MagicMock()

    with (
        _no_real_waiting(),
        patch.object(nodes_module, "logger", logger_double),
    ):
        result = await node(_contextualize_payload())

    chunks = result["contextualized_chunks"]
    assert len(chunks) == 1, "the deterministic-preamble fallback did not run"
    assert chunks[0].preamble, "the fallback produced a chunk with no preamble"

    logged = logger_double.bind.call_args.kwargs["error"]
    assert "quota exceeded for model" in logged
    assert _QuotaExceededError.__name__ in logged


async def test_an_exhausted_retry_reaches_the_entity_extraction_degradation_branch() -> None:
    original = _QuotaExceededError("quota exceeded for model")
    node = make_classify_extract_node(_always_failing_llm(original))
    logger_double = MagicMock()

    with (
        _no_real_waiting(),
        patch.object(nodes_module, "logger", logger_double),
    ):
        result = await node(IngestionState(doc_id=_DOC_ID, contract_metadata=ContractMetadata()))

    # The degradation branch executed: extraction continued with no entities rather than
    # aborting the ingestion.
    assert result["extracted_entities"] == []
    assert result["extracted_relationships"] == []

    logged = logger_double.bind.call_args.kwargs["error"]
    assert "quota exceeded for model" in logged
    assert _QuotaExceededError.__name__ in logged


async def test_the_converted_branch_still_catches_the_unwrapped_framework_failure() -> None:
    """The other route into the same branch, which the conversion must not have closed.

    A deterministic framework failure is outside the boundary's retryable set, so it
    arrives unretried and unwrapped. Converting the `except` to the transient type *alone*
    would have traded one silently-dead degradation branch for another.
    """
    original = LangChainException("model not configured for structured output")
    node = make_segment_document_node(_always_failing_llm(original))

    with _no_real_waiting() as slept:
        result = await node(_segmentation_state())

    assert result.get("segments"), "the fallback segmentation did not run"
    assert slept == [], "a deterministic framework failure must not have been retried"


# --- Proof 4: distinct failure kinds stay distinguishable ---


async def test_authentication_quota_and_malformed_response_do_not_collapse() -> None:
    """One boundary type must not become one boundary *diagnosis*.

    The three kinds reach a caller by two different routes, and both routes preserve
    identity. Quota and malformed-response are retryable, so they arrive wrapped with the
    original as the cause. Authentication is not retryable — waiting does not make
    credentials valid — so it arrives as itself, unwrapped and on the first attempt. Either
    way the caller can tell which of the three happened, which is the property that
    matters; collapsing them into one opaque failure is what the old catch-all did.
    """
    malformed = OutputParserException("expected JSON, got prose")
    quota = _QuotaExceededError("quota exceeded for model")
    unauthorized = _UnauthorizedError("invalid api key")

    observed: dict[str, tuple[type[BaseException], str]] = {}
    for name, error in (
        ("malformed", malformed),
        ("quota", quota),
        ("unauthorized", unauthorized),
    ):

        async def _operation(error: BaseException = error) -> object:
            raise error

        with (
            _no_real_waiting(),
            pytest.raises((TransientExternalError, _UnauthorizedError)) as exc_info,
        ):
            await retry_immediate(_operation, label="gemini_extract_schema", attempts=2)

        raised = exc_info.value
        root = raised.__cause__ if isinstance(raised, TransientExternalError) else raised
        assert root is not None
        observed[name] = (type(root), str(root))

    assert observed["malformed"][0] is OutputParserException
    assert observed["quota"][0] is _QuotaExceededError
    assert observed["unauthorized"][0] is _UnauthorizedError

    # Three distinct types and three distinct messages: nothing was flattened.
    assert len({types for types, _ in observed.values()}) == 3
    assert len({message for _, message in observed.values()}) == 3


async def test_the_retryable_ones_are_wrapped_and_the_unretryable_one_is_not() -> None:
    """The routing itself, asserted separately from the identity it preserves."""
    for error, expected in (
        (OutputParserException("expected JSON, got prose"), TransientExternalError),
        (_QuotaExceededError("quota exceeded for model"), TransientExternalError),
        (_UnauthorizedError("invalid api key"), _UnauthorizedError),
    ):

        async def _operation(error: BaseException = error) -> object:
            raise error

        with _no_real_waiting(), pytest.raises(expected):
            await retry_immediate(_operation, label="gemini_extract_schema", attempts=2)


async def test_the_notes_a_caller_attaches_survive_on_the_transient_wrapper() -> None:
    """A degraded branch annotates what it caught, so the annotation must have a home.

    If the wrapper discarded notes the diagnostic added at the node would vanish, and the
    node's own contribution — which document, which clause, which operation — is the part
    no boundary can reconstruct.
    """
    node = make_contextualize_chunk_node(
        _always_failing_llm(_QuotaExceededError("quota exceeded for model"))
    )
    captured: list[BaseException] = []
    original_describe = nodes_module.describe_failure

    def _capture(exc: BaseException) -> str:
        captured.append(exc)
        return original_describe(exc)

    with _no_real_waiting(), patch.object(nodes_module, "describe_failure", _capture):
        await node(_contextualize_payload())

    assert len(captured) == 1
    note = "\n".join(getattr(captured[0], "__notes__", []))
    assert f"doc_id={_DOC_ID}" in note
    assert f"clause_id={_CLAUSE_ID}" in note
    assert f"chunk_index={_CHUNK_INDEX}" in note
    assert "operation=contextualize" in note
