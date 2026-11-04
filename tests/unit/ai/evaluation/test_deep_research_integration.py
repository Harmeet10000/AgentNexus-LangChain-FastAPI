"""Integration adapters for bounded deep-research evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from app.ai.evaluation.deep_research_integration import (
    DeepResearchInstrumentation,
    spot_check_deep_research_result,
)
from app.ai.evaluation.process_metrics import ProcessMetricsTracker
from app.ai.evaluation.spot_check import RetrievedChunk
from app.features.agent_saul.graph.deep_research import compile_deep_research_subgraph
from app.features.documents.api.http.dto import (
    EvidenceCoverage,
    EvidenceReadItem,
    EvidenceReadResponse,
    EvidenceRef,
    EvidenceSearchItem,
    EvidenceSearchResponse,
)

pytestmark = pytest.mark.unit


class _EvidenceService:
    async def search_evidence(self, **_: object) -> EvidenceSearchResponse:
        return EvidenceSearchResponse(
            status="ok",
            evidence_policy="mechanically_verified",
            items=[
                EvidenceSearchItem(
                    evidence=EvidenceRef(
                        evidence_id="evh-1",
                        chunk_id="chunk-1",
                        document_id="doc-1",
                        document_version=1,
                        page_no=2,
                        graph_indexed=False,
                        retrieval_method="hybrid_rrf",
                        score=0.9,
                        rank=1,
                    ),
                    document_title="Contract",
                    snippet="not retained in graph state",
                    chunk_index=0,
                    chunk_kind="legal_contract",
                )
            ],
            coverage=EvidenceCoverage(documents_requested=1, documents_ready=1),
        )

    async def read_evidence_by_id(self, **_: object) -> EvidenceReadResponse:
        return EvidenceReadResponse(
            status="ok",
            evidence_policy="mechanically_verified",
            items=[
                EvidenceReadItem(
                    evidence=EvidenceRef(
                        evidence_id="evh-1",
                        chunk_id="chunk-1",
                        document_id="doc-1",
                        document_version=1,
                        page_no=2,
                        graph_indexed=False,
                        retrieval_method="handle_read",
                        score=0.0,
                        rank=0,
                    ),
                    document_title="Contract",
                    content="Termination requires thirty days notice.",
                )
            ],
        )


def _initial_state() -> dict[str, object]:
    return {
        "objective": "Termination requires notice",
        "focus": None,
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "allowed_doc_ids": ("doc-1",),
        "thread_id": "thread-1",
        "run_id": "run-1",
        "evidence_checkpoint_id": "run:run-1",
        "queries": [],
        "pending_handles": [],
        "evidence": [],
        "read_notes": [],
        "refusals": [],
        "coverage": {},
        "steps_used": 0,
        "max_steps": 4,
        "max_reads": 4,
        "search_limit": 5,
        "status": "running",
        "summary": None,
        "message": None,
    }


async def test_instrumented_graph_tracks_transitions_tools_and_metrics() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    ticks = count()
    tracker = ProcessMetricsTracker()
    instrumentation = DeepResearchInstrumentation(
        tracker,
        clock=lambda: base + timedelta(seconds=next(ticks)),
    )

    final = await compile_deep_research_subgraph(
        _EvidenceService(), instrumentation=instrumentation
    ).ainvoke(_initial_state())
    snapshot = instrumentation.finish(len(str(final["summary"]).split()))

    assert [transition.node_name for transition in tracker.transitions] == [
        "plan",
        "search",
        "read",
        "synthesize",
    ]
    assert snapshot.total_iterations == 4
    assert sum(len(transition.tool_calls) for transition in tracker.transitions) == 2
    assert snapshot.total_retrieved_documents == 1
    assert snapshot.primary_query_count == 1
    assert snapshot.unique_documents_per_iteration == [0, 1, 0, 0]
    assert snapshot.brief_to_report_coverage_ratio == pytest.approx(
        1 / snapshot.final_report_word_count
    )
    assert snapshot.follow_up_depth == 0
    assert tracker.compute_metrics().total_duration_seconds == 3
    assert snapshot.start_time == base
    assert snapshot.end_time == base + timedelta(seconds=4)


async def test_spot_check_adapter_builds_checks_from_subgraph_state() -> None:
    results = await spot_check_deep_research_result(
        final_state={
            "evidence": [{"evidence_id": "evh-1", "chunk_id": "chunk-1", "document_id": "doc-1"}],
            "read_notes": ["Termination requires notice [evh-1]."],
            "summary": "Termination requires notice.",
        },
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="Termination requires notice.",
            )
        ],
    )

    assert [result.check_name for result in results] == [
        "source_attribution",
        "topic_coverage",
        "missing_citations",
        "citation_density",
        "claim_consistency",
    ]
    assert results[0].passed
    assert results[2].passed
