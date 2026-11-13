"""Checkpoint payload discipline for Agent Saul state (task 6.2).

The checkpoint is the durable copy of a run. Full document text and full clause
text must never enter it: canonical clause text is runtime-only (``exclude=True``
on ``ClauseSegment.text``), the ingestion node no longer writes
``document_text``, and content survives as digests/handles so nodes re-read text
at read time through the server-injected evidence seam.

This suite asserts the *serialized* size limit — the property a reviewer of the
persistence contract actually cares about — on a fully populated, realistically
sized state.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.features.agent_saul.domain.state import (
    ClaimLedgerEntry,
    ClaimSupportStatus,
    ClauseSegment,
    ClauseType,
    EvidenceRef,
    IngestionDocumentState,
    IngestionDocumentStatus,
    PlanActionType,
    PlanStep,
    hydrate_state,
)

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit

#: A distinctive full-clause payload; if any path leaks clause text into the
#: checkpoint, these bytes appear in the serialized state.
CLAUSE_TEXT = (
    "DISTINCTIVE_CLAUSE_PAYLOAD_ the Supplier shall indemnify the Customer "
    "against all losses arising from the breach of section 7.2(b) "
) * 32

SEGMENT_COUNT = 40
REGISTRY_COUNT = 24

#: The serialized checkpoint limit (task 6.2). A state that leaks full clause
#: text would exceed this by more than an order of magnitude (40 segments at
#: ~4 KB each alone is 160 KB of raw text plus envelope overhead).
MAX_SERIALIZED_STATE_BYTES = 256 * 1024


def _segment(index: int) -> ClauseSegment:
    text = f"{CLAUSE_TEXT} [segment {index}]"
    return ClauseSegment(
        clause_id=f"canonical:doc-a:v1:chunk:{index}",
        clause_type=ClauseType.INDEMNITY,
        section_ref=f"7.{index}",
        start_char=0,
        end_char=len(text),
        text=text,
        document_id="doc-a",
        document_version=1,
        chunk_id=f"chunk-{index}",
        chunk_index=index,
        contract_index=0,
        page_no=index + 1,
        metadata={"knowledge_extraction": "complete", "knowledge_extraction_classes": ["PARTY"]},
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
        retrieval_pointer=f"canonical:doc-a:v1:chunk:chunk-{index}",
    )


def _populated_state() -> dict[str, Any]:
    segments = [_segment(index) for index in range(SEGMENT_COUNT)]
    registry = {
        f"evh-{index}": EvidenceRef(
            evidence_id=f"evh-{index}",
            chunk_id=f"chunk-{index % SEGMENT_COUNT}",
            document_id="doc-a",
            document_version=1,
            contract_index=index % 2,
            pipeline_version="canonical-v3",
            page_no=index + 1,
            graph_indexed=index % 3 == 0,
            retrieval_method="hybrid_rrf",
            score=0.9,
            rank=index,
            quote_digest=hashlib.sha256(CLAUSE_TEXT.encode()).hexdigest(),
            quote_digests=(hashlib.sha256(CLAUSE_TEXT.encode()).hexdigest(),) * 4,
            run_id="run-1",
            thread_id="thread-1",
            checkpoint_id="checkpoint-1",
        )
        for index in range(REGISTRY_COUNT)
    }
    return {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "correlation_id": "corr-1",
        "run_id": "run-1",
        "evidence_checkpoint_id": "checkpoint-1",
        "schema_version": 8,
        "doc_id": "doc-a",
        "primary_doc_id": "doc-a",
        "allowed_doc_ids": ["doc-a"],
        "tenant_id": "tenant-1",
        "legal_roles": ["reviewer"],
        "allowed_operations": ["evidence.read"],
        "document_text": None,
        "canonical_segments": segments,
        "canonical_structure_ready": True,
        "legacy_structure_fallback": False,
        "ingestion_state": [
            IngestionDocumentState(document_id="doc-a", status=IngestionDocumentStatus.READY)
        ],
        "messages": [AIMessage(content="analysis envelope", id="m-1")],
        "user_query": "assess indemnity exposure",
        "plan": [PlanStep(step_id="S-1", action=PlanActionType.RISK_ANALYSIS, description="risks")],
        "segments": segments,
        "extracted_entities": [],
        "relationships": [],
        "evidence_registry": registry,
        "claim_ledger": [
            ClaimLedgerEntry(
                claim_id="R-1",
                claim="indemnity is uncapped",
                evidence_ids=["evh-1"],
                contract_index=0,
                support_status=ClaimSupportStatus.SUPPORTED,
            )
        ],
        "coverage": {"documents_requested": 1, "documents_ready": 1},
        "budget_ledger": {
            "limits": {
                "model_calls": 32,
                "retrieval_calls": 30,
                "tool_calls": 50,
                "deep_research_fanout": 4,
                "token_usage": 100000,
                "wall_clock_seconds": 1800,
            },
            "model_calls": 4,
        },
    }


def test_populated_state_serializes_under_the_size_limit_without_clause_text() -> None:
    state = _populated_state()

    _type_tag, payload = JsonPlusSerializer().dumps_typed(state)

    assert CLAUSE_TEXT.encode().split(b"[segment")[0] not in payload
    assert b"DISTINCTIVE_CLAUSE_PAYLOAD_" not in payload
    assert len(payload) < MAX_SERIALIZED_STATE_BYTES


def test_hydrated_legacy_state_defaults_carry_no_document_text() -> None:
    """The oldest supported checkpoint hydrates with no text payload at all."""

    migrated = hydrate_state({"schema_version": 0, "doc_id": "doc-a"})
    assert migrated["document_text"] is None
    assert migrated["canonical_segments"] == []
    _type_tag, payload = JsonPlusSerializer().dumps_typed(migrated)
    assert len(payload) < MAX_SERIALIZED_STATE_BYTES
