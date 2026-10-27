"""Canonical chunk adapter tests for Agent Saul's document boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from returns.result import Success

from app.features.agent_saul.document_source import (
    DOCUMENT_SOURCE_CONFIG_KEY,
    CanonicalDocumentChunk,
)
from app.features.agent_saul.graph.nodes import (
    make_ingestion_node,
    make_normalization_node,
    make_segmentation_node,
)
from app.features.agent_saul.graph.state import (
    ClauseSegment,
    ClauseType,
    WorkflowStatus,
    hydrate_state,
)

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit


def _chunk(doc_id: str, index: int, text: str) -> CanonicalDocumentChunk:
    return CanonicalDocumentChunk(
        chunk_id=f"{doc_id}-chunk-{index}",
        document_id=doc_id,
        document_version=1,
        chunk_index=index,
        content=text,
        page_no=index + 1,
        clause_type="payment",
        locus=f"4.{index + 1}",
    )


class _CanonicalSource:
    def __init__(self, chunks: tuple[CanonicalDocumentChunk, ...]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, ...]] = []

    async def fetch_segments(
        self, *, user_id: str, doc_ids: tuple[str, ...], tenant_id: str | None = None
    ) -> object:
        self.calls.append(doc_ids)
        return Success(self.chunks)


class _LegacySource:
    async def fetch_text(
        self, *, user_id: str, doc_id: str, tenant_id: str | None = None
    ) -> object:
        return Success("legacy document")


def _state(*doc_ids: str) -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "doc_id": doc_ids[0],
        "allowed_doc_ids": list(doc_ids),
        "tenant_id": "tenant-1",
        "retry_count": 0,
        "segments": [],
    }


def test_canonical_clause_text_is_not_checkpoint_serialized() -> None:
    distinctive = "DISTINCTIVE_LEGAL_SOURCE_TEXT_MUST_NOT_LEAK"
    segment = ClauseSegment(
        clause_id="canonical:doc-a:v1:chunk:0",
        clause_type=ClauseType.PAYMENT,
        text=distinctive,
        section_ref="4.1",
        start_char=0,
        end_char=len(distinctive),
        document_id="doc-a",
        document_version=1,
        chunk_id="doc-a-chunk-0",
        content_digest="a" * 64,
        retrieval_pointer="canonical:doc-a:v1:chunk:doc-a-chunk-0",
    )

    payload = JsonPlusSerializer().dumps_typed(
        {"canonical_segments": [segment], "segments": [segment]}
    )[1]

    assert distinctive.encode() not in payload
    assert b"content_digest" in payload
    assert b"retrieval_pointer" in payload


async def test_canonical_ingestion_sorts_documents_and_preserves_duplicate_indices() -> None:
    source = _CanonicalSource(
        (
            _chunk("doc-b", 0, "B first"),
            _chunk("doc-a", 1, "A second"),
            _chunk("doc-a", 0, "A first"),
            _chunk("doc-b", 1, "B second"),
        )
    )

    result = await make_ingestion_node()(
        _state("doc-b", "doc-a"),
        {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: source}},  # type: ignore[arg-type]
    )

    assert result["status"] is WorkflowStatus.NORMALIZING
    assert result["canonical_structure_ready"] is True
    segments = result["segments"]
    assert [segment.document_id for segment in segments] == ["doc-a", "doc-a", "doc-b", "doc-b"]
    assert [segment.chunk_index for segment in segments] == [0, 1, 0, 1]
    assert len({segment.clause_id for segment in segments}) == 4
    assert all(segment.document_id in {"doc-a", "doc-b"} for segment in segments)


async def test_canonical_segment_ids_are_stable_across_runs() -> None:
    chunks = (_chunk("doc-a", 0, "A"), _chunk("doc-b", 0, "B"))
    first = _CanonicalSource(chunks)
    second = _CanonicalSource(chunks)
    node = make_ingestion_node()
    first_result = await node(
        _state("doc-a", "doc-b"), {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: first}}
    )  # type: ignore[arg-type]
    second_result = await node(
        _state("doc-a", "doc-b"), {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: second}}
    )  # type: ignore[arg-type]
    assert [segment.clause_id for segment in first_result["segments"]] == [
        segment.clause_id for segment in second_result["segments"]
    ]


async def test_canonical_path_bypasses_both_structural_llm_nodes() -> None:
    source = _CanonicalSource((_chunk("doc-a", 0, "A"),))
    ingested = await make_ingestion_node()(
        _state("doc-a"),
        {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: source}},  # type: ignore[arg-type]
    )
    state = {**_state("doc-a"), **ingested}

    class _UnexpectedLLM:
        async def ainvoke(self, input: object, config: object = None) -> object:
            message = "canonical chunks must not call a structural LLM"
            raise AssertionError(message)

    normalized = await make_normalization_node(_UnexpectedLLM())(state)  # type: ignore[arg-type]
    state = {**state, **normalized}
    segmented = await make_segmentation_node(_UnexpectedLLM())(state)  # type: ignore[arg-type]
    assert segmented["status"] is WorkflowStatus.EXTRACTING_ENTITIES


async def test_missing_canonical_structure_is_typed_incomplete_for_multi_document_scope() -> None:
    result = await make_ingestion_node()(
        _state("doc-a", "doc-b"),
        {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: _LegacySource()}},  # type: ignore[arg-type]
    )
    assert result["errors"][0].code == "INGESTION_INCOMPLETE"


async def test_legacy_single_document_checkpoint_keeps_explicit_fallback() -> None:
    result = await make_ingestion_node()(
        _state("doc-a"),
        {"configurable": {DOCUMENT_SOURCE_CONFIG_KEY: _LegacySource()}},  # type: ignore[arg-type]
    )
    assert result["legacy_structure_fallback"] is True
    assert result["status"] is WorkflowStatus.NORMALIZING


def test_v3_checkpoint_migrates_with_canonical_state_defaults() -> None:
    migrated = hydrate_state({"schema_version": 3, "doc_id": "doc-a"})
    assert migrated["canonical_segments"] == []
    assert migrated["canonical_structure_ready"] is False
    assert migrated["legacy_structure_fallback"] is False
