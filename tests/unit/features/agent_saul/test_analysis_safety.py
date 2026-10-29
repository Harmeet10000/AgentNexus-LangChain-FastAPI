"""Fail-closed analysis and grounding behavior for Agent Saul."""

import hashlib
from typing import Any

from langchain_core.messages import HumanMessage

from app.features.agent_saul.graph.nodes import (
    make_compliance_node,
    make_finalization_node,
    make_grounding_verification_node,
    make_risk_analysis_node,
)
from app.features.agent_saul.graph.state import (
    AssessmentStatus,
    Citation,
    ComplianceFinding,
    ComplianceOutput,
    ComplianceStatus,
    EvidenceRef,
    FinalReport,
    GroundingStatus,
    GroundingVerificationOutput,
    RiskAnalysisOutput,
    RiskLabel,
    WorkflowStatus,
)


class _Agent:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    async def ainvoke(self, _input: Any) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _state() -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "doc_id": "doc-1",
        "messages": [],
        "segments": [],
        "extracted_entities": [],
        "relationships": [],
        "user_query": "review",
        "status": WorkflowStatus.ANALYZING_RISKS,
    }


def _citation() -> Citation:
    return Citation(claim="the cited claim", source="chunk-1", confidence=0.95)


def _issued_ref(
    *,
    evidence_id: str = "evh-1",
    run_id: str = "run-1",
    graph_indexed: bool = False,
    quote: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        chunk_id="chunk-1",
        document_id="doc-1",
        document_version=1,
        page_no=1,
        graph_indexed=graph_indexed,
        retrieval_method="hybrid_rrf",
        score=0.9,
        rank=1,
        run_id=run_id,
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        quote_digest=(hashlib.sha256(quote.encode()).hexdigest() if quote else None),
    )


def _assessed_state_with_finding(
    citation: Citation,
    *,
    ref: EvidenceRef | None = None,
    coverage: dict[str, int] | None = None,
) -> dict[str, Any]:
    state = _state()
    state.update(
        {
            "run_id": "run-1",
            "evidence_checkpoint_id": "checkpoint-1",
            "allowed_doc_ids": ["doc-1"],
            "evidence_registry": {ref.evidence_id: ref} if ref else {},
            "coverage": coverage or {},
            "risk_analysis": RiskAnalysisOutput(
                findings=[],
                overall_label=RiskLabel.LOW,
                summary="assessed",
            ),
            "compliance_result": ComplianceOutput(
                findings=[
                    ComplianceFinding(
                        finding_id="finding-1",
                        clause_id="c-1",
                        statute="Act s1",
                        is_compliant=True,
                        explanation="supported compliance claim",
                        citations=[citation],
                    )
                ],
                jurisdiction="India",
                summary="assessed",
                compliance_status=ComplianceStatus.COMPLIANT,
            ),
        }
    )
    return state


async def _verified_result(state: dict[str, Any]) -> dict[str, Any]:
    verifier = _Agent(
        GroundingVerificationOutput(
            verified=True,
            unverified_claims=[],
            notes="model said verified",
            citations=[
                Citation(claim="supported compliance claim", source="evh-1", confidence=0.95)
            ],
        )
    )
    return await make_grounding_verification_node(verifier)(state)


async def test_grounding_gate_rejects_forged_evidence_id() -> None:
    state = _assessed_state_with_finding(
        Citation(claim="forged", source="evh-forged", evidence_id="evh-forged", confidence=0.95)
    )
    result = await _verified_result(state)
    assert result["grounding"].verified is False
    assert result["compliance_result"].compliance_status == ComplianceStatus.NOT_ASSESSABLE


async def test_grounding_gate_rejects_evidence_from_another_run() -> None:
    ref = _issued_ref(run_id="run-other")
    state = _assessed_state_with_finding(
        Citation(
            claim="cross-run", source=ref.evidence_id, evidence_id=ref.evidence_id, confidence=0.95
        ),
        ref=ref,
    )
    result = await _verified_result(state)
    assert result["grounding"].verified is False


async def test_grounding_gate_rejects_quote_mismatch() -> None:
    ref = _issued_ref(quote="the actual source text")
    state = _assessed_state_with_finding(
        Citation(
            claim="quote",
            source=ref.evidence_id,
            evidence_id=ref.evidence_id,
            quoted_text="a forged quote",
            confidence=0.95,
        ),
        ref=ref,
    )
    result = await _verified_result(state)
    assert result["grounding"].verified is False


async def test_grounding_gate_rejects_partial_document_coverage() -> None:
    ref = _issued_ref()
    state = _assessed_state_with_finding(
        Citation(
            claim="partial", source=ref.evidence_id, evidence_id=ref.evidence_id, confidence=0.95
        ),
        ref=ref,
        coverage={"documents_requested": 2, "documents_ready": 1, "documents_failed": 1},
    )
    result = await _verified_result(state)
    assert result["grounding"].verified is False


async def test_mechanically_verified_non_graphiti_evidence_can_pass() -> None:
    ref = _issued_ref(graph_indexed=False)
    state = _assessed_state_with_finding(
        Citation(
            claim="supported", source=ref.evidence_id, evidence_id=ref.evidence_id, confidence=0.95
        ),
        ref=ref,
    )
    result = await _verified_result(state)
    assert result["grounding"].verified is True
    assert result["compliance_result"].compliance_status == ComplianceStatus.COMPLIANT


async def test_graph_indexed_only_policy_rejects_non_graphiti_evidence() -> None:
    ref = _issued_ref(graph_indexed=False)
    state = _assessed_state_with_finding(
        Citation(
            claim="not indexed",
            source=ref.evidence_id,
            evidence_id=ref.evidence_id,
            confidence=0.95,
        ),
        ref=ref,
    )
    state["evidence_policy"] = "graph_indexed_only"
    result = await _verified_result(state)
    assert result["grounding"].verified is False


async def test_risk_missing_structured_response_is_not_assessed() -> None:
    result = await make_risk_analysis_node(_Agent({"messages": [HumanMessage(content="plain")]}))(
        _state()
    )

    output = result["risk_analysis"]
    assert output.assessment_status == AssessmentStatus.NOT_ASSESSED
    assert output.overall_label is None
    assert result["errors"][0].code == "RISK_STRUCTURED_OUTPUT_MISSING"


async def test_compliance_agent_failure_is_not_assessable_not_compliant() -> None:
    result = await make_compliance_node(_Agent(error=RuntimeError("tool unavailable")))(_state())

    output = result["compliance_result"]
    assert output.compliance_status == ComplianceStatus.NOT_ASSESSABLE
    assert output.overall_compliant is None
    assert result["status"] == WorkflowStatus.FAILED
    assert result["errors"][0].code == "COMPLIANCE_AGENT_UNAVAILABLE"


async def test_grounding_gate_rejects_positive_llm_verdict_for_unassessed_outputs() -> None:
    state = _state()
    state["risk_analysis"] = RiskAnalysisOutput(
        findings=[],
        summary="not assessed",
        assessment_status=AssessmentStatus.NOT_ASSESSED,
    )
    state["compliance_result"] = ComplianceOutput(
        findings=[],
        jurisdiction="India",
        summary="not assessable",
        compliance_status=ComplianceStatus.NOT_ASSESSABLE,
    )
    verifier = _Agent(
        GroundingVerificationOutput(
            verified=True,
            unverified_claims=[],
            notes="model said verified",
            citations=[_citation()],
        )
    )

    result = await make_grounding_verification_node(verifier)(state)

    assert result["grounding"].verified is False
    assert result["grounding"].status == GroundingStatus.NOT_ASSESSABLE
    assert result["errors"][0].code == "GROUNDING_GATE_REJECTED"


async def test_grounding_invalid_structured_response_is_not_assessable() -> None:
    state = _state()
    verifier = _Agent({"verified": True, "unverified_claims": [], "notes": "missing citations"})

    result = await make_grounding_verification_node(verifier)(state)

    assert result["grounding"].status == GroundingStatus.NOT_ASSESSABLE
    assert result["grounding"].verified is False
    assert result["errors"][0].code == "INVALID_GROUNDING_OUTPUT"


async def test_grounding_gate_downgrades_explicit_unsupported_claims() -> None:
    state = _state()
    state["risk_analysis"] = RiskAnalysisOutput(
        findings=[],
        overall_label=RiskLabel.LOW,
        summary="assessed",
    )
    state["compliance_result"] = ComplianceOutput(
        findings=[],
        jurisdiction="India",
        summary="non-compliant",
        compliance_status=ComplianceStatus.NON_COMPLIANT,
    )
    verifier = _Agent(
        GroundingVerificationOutput(
            verified=True,
            unverified_claims=["unsupported claim"],
            notes="mixed support",
            citations=[_citation()],
        )
    )

    result = await make_grounding_verification_node(verifier)(state)

    assert result["grounding"].verified is False
    assert result["grounding"].status == GroundingStatus.UNVERIFIED
    assert result["errors"][0].code == "GROUNDING_GATE_REJECTED"


async def test_grounding_gate_rejects_compliant_result_with_insufficient_basis() -> None:
    state = _state()
    state["risk_analysis"] = RiskAnalysisOutput(
        findings=[],
        overall_label=RiskLabel.LOW,
        summary="assessed",
    )
    state["compliance_result"] = ComplianceOutput(
        findings=[
            ComplianceFinding(
                finding_id="f-1",
                clause_id="c-1",
                statute="Act s1",
                is_compliant=True,
                explanation="basis missing",
                citations=[_citation()],
                insufficient_basis=True,
            )
        ],
        jurisdiction="India",
        summary="insufficient",
        compliance_status=ComplianceStatus.COMPLIANT,
    )
    verifier = _Agent(
        GroundingVerificationOutput(
            verified=True,
            unverified_claims=[],
            notes="model said verified",
            citations=[_citation()],
        )
    )

    result = await make_grounding_verification_node(verifier)(state)

    assert result["grounding"].verified is False
    assert result["errors"][0].code == "GROUNDING_GATE_REJECTED"


async def test_finalization_does_not_call_model_when_grounding_failed() -> None:
    state = _state()
    state["risk_analysis"] = RiskAnalysisOutput(findings=[], summary="ok")
    state["compliance_result"] = ComplianceOutput(
        findings=[],
        jurisdiction="India",
        summary="not assessable",
        compliance_status=ComplianceStatus.NOT_ASSESSABLE,
    )
    state["grounding"] = GroundingVerificationOutput(
        verified=False,
        status=GroundingStatus.UNVERIFIED,
        unverified_claims=["unsupported"],
        notes="no support",
        citations=[],
    )
    finalizer = _Agent()

    result = await make_finalization_node(finalizer)(state)

    assert finalizer.calls == 0
    assert result["status"] == WorkflowStatus.FAILED
    assert result["errors"][0].code == "GROUNDING_NOT_VERIFIED"
    assert "final_report" not in result


async def test_finalization_marks_report_with_authoritative_statuses() -> None:
    state = _state()
    state["risk_analysis"] = RiskAnalysisOutput(findings=[], summary="risk")
    state["compliance_result"] = ComplianceOutput(
        findings=[],
        jurisdiction="India",
        summary="non-compliant",
        compliance_status=ComplianceStatus.NON_COMPLIANT,
    )
    state["grounding"] = GroundingVerificationOutput(
        verified=True,
        unverified_claims=[],
        notes="verified",
        citations=[_citation()],
    )
    state["human_review"] = type("Review", (), {"approved": True, "overrides": []})()
    report = FinalReport(
        document_id="doc-1",
        summary="report",
        risk_findings=[],
        compliance_findings=[],
        human_overrides=[],
        suggested_actions=[],
        citations=[_citation()],
    )

    result = await make_finalization_node(_Agent(report))(state)

    assert result["final_report"].compliance_status == ComplianceStatus.NON_COMPLIANT
    assert result["final_report"].grounding_status == GroundingStatus.VERIFIED


async def test_finalization_requires_human_review_approval() -> None:
    state = _state()
    state["risk_analysis"] = RiskAnalysisOutput(
        findings=[],
        overall_label=RiskLabel.LOW,
        summary="risk",
    )
    state["compliance_result"] = ComplianceOutput(
        findings=[],
        jurisdiction="India",
        summary="non-compliant",
        compliance_status=ComplianceStatus.NON_COMPLIANT,
    )
    state["grounding"] = GroundingVerificationOutput(
        verified=True,
        unverified_claims=[],
        notes="verified",
        citations=[_citation()],
    )

    finalizer = _Agent()
    result = await make_finalization_node(finalizer)(state)

    assert finalizer.calls == 0
    assert result["status"] == WorkflowStatus.FAILED
    assert result["errors"][0].code == "HUMAN_REVIEW_NOT_APPROVED"
