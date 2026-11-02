"""D7: the verdict path emits shared-vocabulary outcomes; behaviour unchanged.

The mapping helpers are type-level: the gate bodies are the ones hardened in
6.5 and their existing suite (``test_analysis_safety.py``) is the acceptance
gate. These tests prove the shared-vocabulary shapes are reachable from the
real gates — ACCEPT and ESCALATE both — and that a mechanical finding renders
as a semantic-report finding with its locator.
"""

from typing import Any

import pytest

from app.ai.validation import SemanticReport, ValidationDecision, ValidationReason
from app.features.agent_saul.domain.state import (
    Citation,
    ComplianceFinding,
    ComplianceOutput,
    ComplianceStatus,
    EvidenceRef,
    RiskAnalysisOutput,
    RiskLabel,
    WorkflowStatus,
)
from app.features.agent_saul.graph.nodes import (
    GroundingVerificationOutput,
    _finding_evidence_issues,
    make_grounding_verification_node,
    verdict_issue_finding,
    verdict_mechanical_report,
    verdict_outcome,
)

pytestmark = pytest.mark.unit


class _Agent:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    async def ainvoke(self, _input: Any) -> Any:
        if self.error is not None:
            raise self.error
        return self.result


def _state() -> dict[str, Any]:
    return {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "doc_id": "doc-1",
        "run_id": "run-1",
        "evidence_checkpoint_id": "checkpoint-1",
        "allowed_doc_ids": ["doc-1"],
        "messages": [],
        "segments": [],
        "extracted_entities": [],
        "relationships": [],
        "user_query": "review",
        "status": WorkflowStatus.ANALYZING_RISKS,
    }


def _issued_ref(quote: str | None = None) -> EvidenceRef:
    import hashlib

    return EvidenceRef(
        evidence_id="evh-1",
        chunk_id="chunk-1",
        document_id="doc-1",
        document_version=1,
        page_no=1,
        graph_indexed=False,
        retrieval_method="hybrid_rrf",
        score=0.9,
        rank=1,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_id="checkpoint-1",
        quote_digest=(hashlib.sha256(quote.encode()).hexdigest() if quote else None),
    )


def _assessed_state(citation: Citation, *, ref: EvidenceRef | None = None) -> dict[str, Any]:
    state = _state()
    state["evidence_registry"] = {ref.evidence_id: ref} if ref else {}
    state["risk_analysis"] = RiskAnalysisOutput(
        findings=[],
        overall_label=RiskLabel.LOW,
        summary="assessed",
    )
    state["compliance_result"] = ComplianceOutput(
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
    )
    return state


def _verified_verifier() -> _Agent:
    return _Agent(
        GroundingVerificationOutput(
            verified=True,
            unverified_claims=[],
            notes="model said verified",
            citations=[
                Citation(claim="supported compliance claim", source="evh-1", confidence=0.95)
            ],
        )
    )


async def test_accept_outcome_is_reachable_through_the_real_path() -> None:
    ref = _issued_ref()
    state = _assessed_state(
        Citation(
            claim="supported", source=ref.evidence_id, evidence_id=ref.evidence_id, confidence=0.9
        ),
        ref=ref,
    )
    result = await make_grounding_verification_node(_verified_verifier())(state)

    assert result["grounding"].verified is True
    outcome = verdict_outcome(result["grounding"])
    assert outcome.decision is ValidationDecision.ACCEPT
    assert outcome.reason is ValidationReason.ACCEPTED
    assert outcome.value is result["grounding"]


async def test_escalate_outcome_is_reachable_through_the_real_path() -> None:
    state = _assessed_state(
        Citation(claim="forged", source="evh-x", evidence_id="evh-x", confidence=0.9)
    )
    result = await make_grounding_verification_node(_verified_verifier())(state)

    assert result["grounding"].verified is False
    error_code = result["errors"][0].code
    outcome = verdict_outcome(result["grounding"], error_code=error_code)
    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.NON_RETRYABLE_REJECTION
    assert outcome.value is None
    assert outcome.semantic is not None
    assert outcome.semantic.verdict == "reject"
    assert outcome.semantic.findings


async def test_unavailable_verifier_escalates_as_evaluator_unavailable() -> None:
    state = _assessed_state(
        Citation(claim="c", source="evh-1", evidence_id="evh-1", confidence=0.9)
    )
    result = await make_grounding_verification_node(_Agent(error=RuntimeError("down")))(state)

    assert result["errors"][0].code == "GROUNDING_VERIFIER_UNAVAILABLE"
    outcome = verdict_outcome(result["grounding"], error_code=result["errors"][0].code)
    assert outcome.decision is ValidationDecision.ESCALATE
    assert outcome.reason is ValidationReason.EVALUATOR_UNAVAILABLE
    assert outcome.semantic is not None
    assert outcome.semantic.verdict == "unavailable"


def test_mechanical_gate_finding_renders_with_its_locator() -> None:
    """1.2: a real mechanical gate issue becomes a finding with a locator."""
    state = _state()
    state["evidence_registry"] = {}
    issues = _finding_evidence_issues(
        state,
        [Citation(claim="c", source="evh-unknown", evidence_id="evh-unknown", confidence=0.9)],
        "finding-1",
    )

    assert issues == ["Finding finding-1 cites an unknown or forged evidence handle."]
    finding = verdict_issue_finding(issues[0])
    assert finding.code == "GROUNDING_GATE_REJECTED"
    assert finding.message == issues[0]
    assert finding.locator == "finding-1"


def test_quote_span_issue_renders_with_the_evidence_handle_locator() -> None:
    ref = _issued_ref(quote="the actual source text")
    state = _state()
    state["evidence_registry"] = {ref.evidence_id: ref}
    issues = _finding_evidence_issues(
        state,
        [
            Citation(
                claim="quote",
                source=ref.evidence_id,
                evidence_id=ref.evidence_id,
                quoted_text="a forged quote",
                confidence=0.9,
            )
        ],
        "finding-1",
    )

    assert issues == ["Finding finding-1 has a quote that does not match read evidence."]
    finding = verdict_issue_finding(issues[0])
    assert finding.locator == "finding-1"


def test_mechanical_report_passes_only_without_findings() -> None:
    assert verdict_mechanical_report([]) == SemanticReport(verdict="pass")

    report = verdict_mechanical_report(
        ["Quote citation evh-1 is not a span of its canonical chunk."]
    )
    assert report.verdict == "reject"
    assert report.findings[0].locator == "evh-1"
    assert report.findings[0].code == "GROUNDING_GATE_REJECTED"
