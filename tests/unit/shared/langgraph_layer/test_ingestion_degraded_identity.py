"""Unit tests for task A5 — a fanned-out node's degradation record carries document identity.

A5's stated premise is refuted, and the amendment in `tasks.md` records why: the node
the task names builds its note from a local validated `ClauseSegment`, and the two nodes
that *do* read `state.doc_id` receive `IngestionState`, a Pydantic model whose `doc_id` is
a defaulted field — so no attribute access in any handler can raise.

What A5's Proof 2(c) exposes is real and smaller. `dispatch_contextualize_chunks` built
its `Send` payload without `doc_id`, and because `Send` *replaces* the state for the
fanned-out invocation, there was no document identity reachable from that node at all. A
degraded contextualization was attributable to a clause but never to a document, which
`clause_id` alone does not fix under concurrent ingestion. These tests pin the payload and
the diagnostic.

Proof 2(a) and 2(b) — that the branch returns a degraded result rather than raising, and
that the diagnostic names the original cause — are **deferred to C6**, and
`test_the_boundary_still_converts_the_type_the_handler_catches` below records exactly why:
`retry_immediate` converts every failure to `TransientExternalError`, which
`except LangChainException` cannot match, so in production this branch is unreachable. The
tests here reach it by patching that boundary, which isolates the handler under test and
keeps the boundary's own defect where C6 can fix it once for all three call sites.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.exceptions import LangChainException

from app.shared.langgraph_layer.ingestion_kb import nodes as nodes_module
from app.shared.langgraph_layer.ingestion_kb.nodes import (
    dispatch_contextualize_chunks,
    make_contextualize_chunk_node,
)
from app.shared.langgraph_layer.ingestion_kb.state import (
    ClauseSegment,
    IngestionState,
)
from app.shared.langgraph_layer.kb_retry import TransientExternalError, retry_immediate

_DOC_ID = "doc-42"
_CLAUSE_ID = "clause-7"
_CHUNK_INDEX = 3


def _segment(clause_id: str = _CLAUSE_ID, chunk_index: int = _CHUNK_INDEX) -> ClauseSegment:
    return ClauseSegment(
        clause_id=clause_id,
        text="The Supplier shall indemnify the Customer against all third-party claims.",
        chunk_index=chunk_index,
        page_no=2,
    )


def _payload(**overrides: Any) -> dict[str, Any]:
    """A `Send` payload shaped the way the dispatcher builds one."""
    base: dict[str, Any] = {
        "doc_id": _DOC_ID,
        "segment": _segment().model_dump(),
        "contract_metadata": {},
        "source": "upload",
    }
    return base | overrides


def _raising_boundary(error: Exception) -> Any:
    """Stand in for `retry_immediate`, raising `error` without converting its type.

    Patching here rather than making the language-model double fail is deliberate: the
    real boundary converts every exception to `TransientExternalError` (C6), so a test
    that failed the model would never reach the handler. Isolating the handler is the
    only way to test it before C6 lands, and it leaves the conversion defect in one
    place instead of working around it three times.
    """

    async def _boundary(_operation: Any, *, label: str) -> Any:  # noqa: ARG001 — signature parity
        raise error

    return _boundary


# --- The dispatcher carries document identity into the fan-out ---


def test_dispatcher_puts_doc_id_in_every_send_payload() -> None:
    state = IngestionState(doc_id=_DOC_ID, segments=[_segment("a", 0), _segment("b", 1)])

    sends = dispatch_contextualize_chunks(state)

    assert len(sends) == 2
    assert [send.arg["doc_id"] for send in sends] == [_DOC_ID, _DOC_ID]


def test_dispatcher_send_payload_keys_are_exactly_what_the_node_reads() -> None:
    """Guards against the inverse of the original defect — an unread key added later.

    `Send` replaces the state, so this dict is the node's entire world. Anything absent
    is unreachable from the node and anything present but unread is dead weight that
    reads like a contract.
    """
    state = IngestionState(doc_id=_DOC_ID, segments=[_segment()])

    assert set(dispatch_contextualize_chunks(state)[0].arg) == {
        "doc_id",
        "segment",
        "contract_metadata",
        "source",
    }


def test_dispatcher_targets_the_contextualize_node() -> None:
    state = IngestionState(doc_id=_DOC_ID, segments=[_segment()])

    assert dispatch_contextualize_chunks(state)[0].node == "contextualize_chunks"


def test_dispatcher_fans_out_one_send_per_segment() -> None:
    segments = [_segment(f"clause-{index}", index) for index in range(5)]
    state = IngestionState(doc_id=_DOC_ID, segments=segments)

    assert len(dispatch_contextualize_chunks(state)) == 5


# --- A5 Proof 2(c): the degradation record names document AND chunk ---


async def test_degraded_note_carries_document_and_chunk_identity() -> None:
    provider_error = LangChainException("structured output refused")
    node = make_contextualize_chunk_node(MagicMock())

    with patch.object(nodes_module, "retry_immediate", _raising_boundary(provider_error)):
        result = await node(_payload())

    # (a) locally: degraded, not raised. In production this is C6-gated — see the
    # boundary test below.
    assert "contextualized_chunks" in result

    note = "\n".join(provider_error.__notes__)
    assert f"doc_id={_DOC_ID}" in note
    assert f"clause_id={_CLAUSE_ID}" in note
    assert f"chunk_index={_CHUNK_INDEX}" in note
    assert "operation=contextualize" in note


async def test_degraded_chunk_is_the_deterministic_preamble_not_a_placeholder() -> None:
    """The fallback must be a real chunk, not an empty one that indexes and ranks nothing.

    Same failure shape A2 removed from the embedder: a substituted value that inserts
    cleanly and retrieves nothing is worse than an error, because nothing reports it.
    """
    node = make_contextualize_chunk_node(MagicMock())
    segment = _segment()

    with patch.object(
        nodes_module, "retry_immediate", _raising_boundary(LangChainException("refused"))
    ):
        result = await node(_payload())

    chunks = result["contextualized_chunks"]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.clause_id == _CLAUSE_ID
    assert chunk.chunk_index == _CHUNK_INDEX
    assert chunk.text == segment.text
    assert chunk.preamble
    assert chunk.tokens > 0


async def test_a_payload_without_doc_id_degrades_without_raising() -> None:
    """The defensive read is the point of A5, applied to A5's own fix.

    `state["doc_id"]` would turn a recoverable contextualization failure into a
    `KeyError` raised *inside the handler*, replacing the original diagnostic — the exact
    defect A5 set out to remove, reintroduced by the naive version of its own fix.
    """
    provider_error = LangChainException("refused")
    node = make_contextualize_chunk_node(MagicMock())
    payload = _payload()
    del payload["doc_id"]

    with patch.object(nodes_module, "retry_immediate", _raising_boundary(provider_error)):
        result = await node(payload)

    assert "contextualized_chunks" in result
    assert "doc_id=," in "\n".join(provider_error.__notes__)


async def test_the_success_path_is_unchanged_by_the_identity_fix() -> None:
    node = make_contextualize_chunk_node(MagicMock())
    returned = {
        "clause_id": _CLAUSE_ID,
        "chunk_index": _CHUNK_INDEX,
        "preamble": "Indemnity — Supplier obligations",
        "text": "The Supplier shall indemnify the Customer.",
        "tokens": 9,
    }

    async def _boundary(_operation: Any, *, label: str) -> Any:  # noqa: ARG001 — signature parity
        return returned

    with patch.object(nodes_module, "retry_immediate", _boundary):
        result = await node(_payload())

    chunk = result["contextualized_chunks"][0]
    assert chunk.preamble == "Indemnity — Supplier obligations"
    assert chunk.tokens == 9


# --- Why 2(a) and 2(b) are deferred, pinned so C6 must revisit it ---


async def test_the_boundary_still_converts_the_type_the_handler_catches() -> None:
    """C6's defect, confirmed — and broader than C6 states.

    `retry_immediate` catches `Exception` and raises `TransientExternalError from exc`.
    Every degraded branch in `nodes.py` catches `LangChainException`, which is not a base
    of `TransientExternalError`, so **none** of the three branches can fire in production:
    segmentation, contextualize, and entity extraction alike. The pipeline propagates
    where it appears to degrade.

    This test fails once C6 makes the boundary raise a type the callers catch, which is
    the intent — A5's Proof 2(a)/2(b) belong to the test C6's third Proof already asks
    for, and this is the tripwire that forces it to be written.
    """
    original = LangChainException("provider refused")

    async def _always_fails() -> object:
        raise original

    with pytest.raises(TransientExternalError) as exc_info:
        await retry_immediate(_always_fails, label="contextualize_chunk", attempts=1)

    # The original survives as the cause, which is all `reraise=True` buys here.
    assert exc_info.value.__cause__ is original
    # But the type the handlers catch does not survive, which is the whole problem.
    assert not isinstance(exc_info.value, LangChainException)
