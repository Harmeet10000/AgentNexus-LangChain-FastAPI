"""Deep-research plan execution stays scoped and metadata-only."""

from typing import Any

import pytest

from app.features.agent_saul.graph.nodes import make_deep_research_node
from app.features.agent_saul.graph.state import PlanActionType, PlanStep, WorkflowStatus
from app.features.documents.dto import (
    EvidenceCoverage,
    EvidenceRef,
    EvidenceSearchItem,
    EvidenceSearchResponse,
)

pytestmark = pytest.mark.unit


def _step(step_id: str = "research-1", description: str = "first question") -> PlanStep:
    return PlanStep(
        step_id=step_id, action=PlanActionType.SEARCH_PRECEDENTS, description=description
    )


def _state(*, plan: list[PlanStep], current_step: int = 0) -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "thread_id": "thread-1",
        "run_id": "run-1",
        "evidence_checkpoint_id": "checkpoint-1",
        "allowed_doc_ids": ["doc-1", "doc-2"],
        "plan": plan,
        "current_step": current_step,
        "evidence_registry": {},
        "coverage": {},
        "errors": [],
        "status": WorkflowStatus.PLAN_APPROVED,
        "evidence_policy": "mechanically_verified",
    }


def _response(
    *,
    status: str = "ok",
    evidence_id: str = "evh-1",
    coverage: EvidenceCoverage | None = None,
    message: str | None = None,
) -> EvidenceSearchResponse:
    items: list[EvidenceSearchItem] = []
    if status == "ok":
        ref = EvidenceRef(
            evidence_id=evidence_id,
            chunk_id="chunk-1",
            document_id="doc-1",
            document_version=1,
            page_no=3,
            graph_indexed=False,
            retrieval_method="hybrid_rrf",
            score=0.9,
            rank=1,
        )
        items.append(
            EvidenceSearchItem(
                evidence=ref,
                document_title="Uploaded authority",
                snippet="this snippet must not enter state",
                chunk_index=0,
                chunk_kind="generic",
            )
        )
    return EvidenceSearchResponse(
        status=status,  # type: ignore[arg-type]
        evidence_policy="mechanically_verified",
        items=items,
        coverage=coverage or EvidenceCoverage(documents_requested=2, documents_ready=2),
        message=message,
    )


class _EvidenceService:
    def __init__(self, response: EvidenceSearchResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def search_evidence(self, **kwargs: Any) -> EvidenceSearchResponse:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


async def test_plan_without_research_does_not_call_service() -> None:
    service = _EvidenceService(_response())
    state = _state(
        plan=[PlanStep(step_id="summary-1", action=PlanActionType.SUMMARIZE, description="report")]
    )

    result = await make_deep_research_node(evidence_service=service)(state)  # type: ignore[arg-type]

    assert result == {}
    assert service.calls == []


async def test_scoped_research_issues_metadata_without_content() -> None:
    service = _EvidenceService(_response())
    result = await make_deep_research_node(evidence_service=service)(
        _state(plan=[_step()])  # type: ignore[arg-type]
    )

    assert service.calls[0]["principal_id"] == "user-1"
    assert service.calls[0]["tenant_id"] == "tenant-1"
    assert service.calls[0]["allowed_doc_ids"] == ("doc-1", "doc-2")
    assert result["evidence_registry"]["evh-1"].document_id == "doc-1"
    assert "this snippet" not in result["deep_research_results"]
    assert "evh-1" in result["deep_research_results"]
    assert result["current_step"] == 1


async def test_unavailable_evidence_service_is_typed_and_terminal() -> None:
    service = _EvidenceService(RuntimeError("dependency unavailable"))
    result = await make_deep_research_node(evidence_service=service)(
        _state(plan=[_step()])  # type: ignore[arg-type]
    )

    assert result["deep_research_results"] == "[]"
    assert result["status"] is WorkflowStatus.FAILED
    assert result["errors"][0].code == "DEEP_RESEARCH_SERVICE_UNAVAILABLE"


async def test_missing_evidence_service_is_not_silent() -> None:
    result = await make_deep_research_node()(_state(plan=[_step()]))  # type: ignore[arg-type]

    assert result["status"] is WorkflowStatus.FAILED
    assert result["errors"][0].code == "DEEP_RESEARCH_SERVICE_UNAVAILABLE"


async def test_partial_coverage_is_preserved_as_warning() -> None:
    service = _EvidenceService(
        _response(
            coverage=EvidenceCoverage(
                documents_requested=2,
                documents_ready=1,
                documents_failed=1,
            )
        )
    )
    result = await make_deep_research_node(evidence_service=service)(
        _state(plan=[_step()])  # type: ignore[arg-type]
    )

    assert result["evidence_registry"]
    assert result["coverage"]["documents_failed"] == 1
    assert result["errors"][0].code == "DEEP_RESEARCH_PARTIAL_COVERAGE"
    assert result["status"] is WorkflowStatus.PLAN_APPROVED


async def test_no_evidence_is_not_assessable() -> None:
    service = _EvidenceService(
        _response(
            status="no_evidence",
            message="No uploaded authority matched.",
            coverage=EvidenceCoverage(documents_requested=2, documents_ready=2),
        )
    )
    result = await make_deep_research_node(evidence_service=service)(
        _state(plan=[_step()])  # type: ignore[arg-type]
    )

    assert result["status"] is WorkflowStatus.FAILED
    assert result["errors"][0].code == "DEEP_RESEARCH_NO_EVIDENCE"
