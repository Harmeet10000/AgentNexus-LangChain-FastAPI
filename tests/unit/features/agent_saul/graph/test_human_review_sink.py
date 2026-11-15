"""D7 task 2.1: an ESCALATE outcome reaches the HITL interrupt through the sink.

The graph below wires the real review node exactly as the factory does — via
``make_human_review_entry``, whose sink is the only way an escalation gets in.
The interrupt fires once, through the sink, and the resume (approval) flow is
the existing HITL flow, unchanged.
"""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.ai.validation import EvaluationContext, ValidationDecision, ValidationReason
from app.features.agent_saul.domain.state import (
    Citation,
    ComplianceFinding,
    ComplianceOutput,
    ComplianceStatus,
    GroundingStatus,
    GroundingVerificationOutput,
    LegalAgentState,
    RiskAnalysisOutput,
    RiskLabel,
    WorkflowStatus,
)
from app.features.agent_saul.graph.nodes import (
    make_grounding_verification_node,
    make_human_review_entry,
    verdict_outcome,
)

pytestmark = pytest.mark.unit

_SEAM = "agent_saul.verdict"


def _escalated_outcome() -> Any:
    """A review-class ESCALATE outcome from the verdict path's own mapping."""
    return verdict_outcome(
        GroundingVerificationOutput(
            verified=False,
            status=GroundingStatus.UNVERIFIED,
            unverified_claims=["Quote citation evh-1 is not a span of its canonical chunk."],
            notes="gate rejected",
            citations=[],
        )
    )


def _state() -> dict[str, Any]:
    return {
        "user_id": "analyst-1",
        "doc_id": "doc-1",
        "messages": [],
        "segments": [],
        "extracted_entities": [],
        "relationships": [],
        "user_query": "review",
        "evidence_registry": {},
        "coverage": {},
        "evidence_policy": "mechanically_verified",
        "status": WorkflowStatus.AWAITING_HUMAN_REVIEW,
    }


def _wired_graph() -> tuple[Any, Any, list[ValidationDecision]]:
    """A real graph: validate escalates through the sink, then the review node runs."""
    sink, review_node = make_human_review_entry()
    escalated = _escalated_outcome()
    delivered: list[ValidationDecision] = []

    async def deliver_escalation(state: LegalAgentState) -> dict[str, Any]:
        await sink.escalate(escalated, EvaluationContext(seam=_SEAM))
        delivered.append(escalated.decision)
        return {}

    builder = StateGraph(LegalAgentState)
    builder.add_node("validate", deliver_escalation)
    builder.add_node("human_review", review_node)
    builder.add_edge(START, "validate")
    builder.add_edge("validate", "human_review")
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=InMemorySaver()), sink, delivered


async def test_escalate_interrupt_fires_once_through_the_sink() -> None:
    graph, _sink, delivered = _wired_graph()
    config = {"configurable": {"thread_id": "thread-1"}}

    paused = await graph.ainvoke(_state(), config)

    # The interrupt fired exactly once, and its payload arrived via the sink:
    # the escalation reason and findings ride the review payload.
    assert delivered == [ValidationDecision.ESCALATE]
    payload = paused["__interrupt__"][0].value
    assert len(payload["escalations"]) == 1
    escalation = payload["escalations"][0]
    assert escalation["decision"] == "escalate"
    assert escalation["reason"] == ValidationReason.NON_RETRYABLE_REJECTION.value
    assert escalation["findings"][0]["locator"] == "evh-1"
    # The rest of the payload is the existing review payload, unchanged.
    assert payload["type"] == "human_review_required"
    assert payload["grounding_verified"] is False


async def test_resume_through_the_sink_driven_interrupt_is_the_existing_flow() -> None:
    graph, _sink, _delivered = _wired_graph()
    config = {"configurable": {"thread_id": "thread-1"}}

    await graph.ainvoke(_state(), config)
    resumed = await graph.ainvoke(
        Command(
            resume={
                "action": "approve",
                "reviewer_id": "reviewer-1",
                "reviewer_role": "reviewer",
            }
        ),
        config,
    )

    # The approval/resume semantics are exactly the existing HITL flow.
    assert resumed["human_review"].approved is True
    assert resumed["human_review"].reviewer_id == "reviewer-1"
    assert resumed["self_reviewed"] is False
    assert resumed["status"] == WorkflowStatus.FINALIZING


async def test_without_an_escalation_the_payload_is_the_pre_adoption_one() -> None:
    _sink, review_node = make_human_review_entry()

    async def no_escalation(state: LegalAgentState) -> dict[str, Any]:
        return {}

    builder = StateGraph(LegalAgentState)
    builder.add_node("validate", no_escalation)
    builder.add_node("human_review", review_node)
    builder.add_edge(START, "validate")
    builder.add_edge("validate", "human_review")
    builder.add_edge("human_review", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    paused = await graph.ainvoke(_state(), {"configurable": {"thread_id": "thread-2"}})

    assert "escalations" not in paused["__interrupt__"][0].value


def _escalated_outcome_for(locator: str) -> Any:
    """A review-class ESCALATE outcome whose finding is identifiable."""
    return verdict_outcome(
        GroundingVerificationOutput(
            verified=False,
            status=GroundingStatus.UNVERIFIED,
            unverified_claims=[f"Quote citation {locator} is not a span of its canonical chunk."],
            notes="gate rejected",
            citations=[],
        )
    )


def _wired_two_thread_graph() -> Any:
    """The factory wiring, with each thread escalating its own finding."""
    sink, review_node = make_human_review_entry()

    async def deliver_for_this_thread(state: LegalAgentState) -> dict[str, Any]:
        await sink.escalate(_escalated_outcome_for(state["doc_id"]), EvaluationContext(seam=_SEAM))
        return {}

    builder = StateGraph(LegalAgentState)
    builder.add_node("validate", deliver_for_this_thread)
    builder.add_node("human_review", review_node)
    builder.add_edge(START, "validate")
    builder.add_edge("validate", "human_review")
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=InMemorySaver())


async def test_escalations_never_leak_across_threads() -> None:
    """Thread B's interrupt payload carries only thread B's escalations.

    The sink buffers per LangGraph thread and the review node drains only the
    current thread's deliveries, so thread A's escalated findings can never
    surface in thread B's review.
    """
    graph = _wired_two_thread_graph()

    paused_a = await graph.ainvoke(
        {**_state(), "doc_id": "evh-A"},
        {"configurable": {"thread_id": "thread-A"}},
    )
    payload_a = paused_a["__interrupt__"][0].value
    assert [e["findings"][0]["locator"] for e in payload_a["escalations"]] == ["evh-A"]

    paused_b = await graph.ainvoke(
        {**_state(), "doc_id": "evh-B"},
        {"configurable": {"thread_id": "thread-B"}},
    )
    payload_b = paused_b["__interrupt__"][0].value
    locators = [e["findings"][0]["locator"] for e in payload_b["escalations"]]
    assert locators == ["evh-B"]
    assert "evh-A" not in locators


async def test_checkpointed_channel_carries_escalations_without_the_sink() -> None:
    """The ``pending_escalations`` state channel renders and drains.

    A validating node that returns its collected ESCALATE outcomes as its
    ``pending_escalations`` state update gives the review node carriage that
    survives worker migration and restarts — no process-local delivery
    involved — and the review node clears the channel once rendered.
    """
    _sink, review_node = make_human_review_entry()
    outcome = _escalated_outcome_for("evh-9")

    async def deliver_via_state(state: LegalAgentState) -> dict[str, Any]:
        return {"pending_escalations": [outcome]}

    builder = StateGraph(LegalAgentState)
    builder.add_node("validate", deliver_via_state)
    builder.add_node("human_review", review_node)
    builder.add_edge(START, "validate")
    builder.add_edge("validate", "human_review")
    builder.add_edge("human_review", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-3"}}

    paused = await graph.ainvoke(_state(), config)

    payload = paused["__interrupt__"][0].value
    assert payload["escalations"][0]["reason"] == ValidationReason.NON_RETRYABLE_REJECTION.value
    assert payload["escalations"][0]["findings"][0]["locator"] == "evh-9"

    resumed = await graph.ainvoke(
        Command(resume={"action": "approve", "reviewer_id": "r-1", "reviewer_role": "reviewer"}),
        config,
    )

    assert resumed["pending_escalations"] == []
    assert resumed["human_review"].approved is True


# ---------------------------------------------------------------------------
# The factory wiring: the real verdict node escalates through the sink


class _VerifiedVerifier:
    """A grounding verifier that claims verified; the mechanical gate catches it."""

    async def ainvoke(self, _input: Any) -> Any:
        return GroundingVerificationOutput(
            verified=True,
            unverified_claims=[],
            notes="model said verified",
            citations=[Citation(claim="forged", source="evh-x", confidence=0.95)],
        )


def _assessed_state_with_forged_handle() -> dict[str, Any]:
    state = _state()
    state["risk_analysis"] = RiskAnalysisOutput(
        findings=[], overall_label=RiskLabel.LOW, summary="assessed"
    )
    state["compliance_result"] = ComplianceOutput(
        findings=[
            ComplianceFinding(
                finding_id="finding-1",
                clause_id="c-1",
                statute="Act s1",
                is_compliant=True,
                explanation="supported compliance claim",
                citations=[Citation(claim="forged", source="evh-x", confidence=0.9)],
            )
        ],
        jurisdiction="India",
        summary="assessed",
        compliance_status=ComplianceStatus.COMPLIANT,
    )
    return state


def _wired_factory_graph() -> Any:
    """The real grounding-verification node wired with its sink, then review."""
    sink, review_node = make_human_review_entry()

    builder = StateGraph(LegalAgentState)
    builder.add_node(
        "grounding_verification",
        make_grounding_verification_node(_VerifiedVerifier(), human_review_sink=sink),
    )
    builder.add_node("human_review", review_node)
    builder.add_edge(START, "grounding_verification")
    builder.add_edge("grounding_verification", "human_review")
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=InMemorySaver())


async def test_grounding_failure_escalates_through_the_factory_wired_sink() -> None:
    """A forged evidence handle fails the grounding gate and reaches HITL.

    The verdict node's ESCALATE outcome is delivered through the sink's
    thread-buffered enqueue (the path the review node drains), so the forged
    citation surfaces in that thread's interrupt payload — while the node's
    returned state stays the existing verdict shape.
    """
    graph = _wired_factory_graph()
    config = {"configurable": {"thread_id": "thread-verdict"}}

    paused = await graph.ainvoke(_assessed_state_with_forged_handle(), config)

    payload = paused["__interrupt__"][0].value
    assert len(payload["escalations"]) == 1
    escalation = payload["escalations"][0]
    assert escalation["decision"] == "escalate"
    assert escalation["reason"] == ValidationReason.NON_RETRYABLE_REJECTION.value
    assert escalation["findings"][0]["locator"] == "finding-1"
    # The verdict node's own state update is the pre-existing verdict shape —
    # the sink, not a state channel, carried the escalation.
    assert paused["grounding"].verified is False
    assert [error.code for error in paused["errors"]] == ["GROUNDING_GATE_REJECTED"]


async def test_verdict_escalations_never_leak_across_threads() -> None:
    """A clean thread's review payload carries no other thread's escalation.

    Thread A's forged-handle grounding failure escalates through the sink;
    thread B's verified grounding run produces no ESCALATE outcome, so its
    payload is the pre-adoption review payload.
    """
    graph = _wired_factory_graph()

    paused_a = await graph.ainvoke(
        _assessed_state_with_forged_handle(),
        {"configurable": {"thread_id": "thread-A"}},
    )
    payload_a = paused_a["__interrupt__"][0].value
    assert payload_a["escalations"][0]["findings"][0]["locator"] == "finding-1"

    clean = _assessed_state_with_forged_handle()
    clean["compliance_result"] = ComplianceOutput(
        findings=[],
        jurisdiction="India",
        summary="assessed",
        compliance_status=ComplianceStatus.COMPLIANT,
    )
    paused_b = await graph.ainvoke(clean, {"configurable": {"thread_id": "thread-B"}})

    payload_b = paused_b["__interrupt__"][0].value
    assert "escalations" not in payload_b
    assert payload_b["grounding_verified"] is True


# ---------------------------------------------------------------------------
# Task 6.7: rendering of unsupported/contradicted/not-assessed items and
# override recording (reason, identity, self_reviewed, contract identity).
# ---------------------------------------------------------------------------


def _review_state() -> dict[str, Any]:
    from app.features.agent_saul.domain.state import ClauseSegment, ClauseType

    state = _state()
    state["segments"] = [
        ClauseSegment(
            clause_id="canonical:doc-a:v1:chunk:1",
            clause_type=ClauseType.INDEMNITY,
            section_ref="7.1",
            start_char=0,
            end_char=10,
            document_id="doc-a",
            document_version=1,
            chunk_id="chunk-1",
            chunk_index=1,
            contract_index=0,
        ),
        ClauseSegment(
            clause_id="canonical:doc-a:v1:chunk:2",
            clause_type=ClauseType.TERMINATION,
            section_ref="12.1",
            start_char=0,
            end_char=10,
            document_id="doc-a",
            document_version=1,
            chunk_id="chunk-2",
            chunk_index=2,
            contract_index=1,
        ),
    ]
    return state


async def test_review_payload_renders_unsupported_contradicted_and_not_assessed() -> None:
    """Every item a reviewer approves that is not machine-supported is rendered."""
    from app.features.agent_saul.domain.state import (
        Citation,
        ClaimLedgerEntry,
        ClaimSupportStatus,
        ComplianceStatus,
        RiskFinding,
    )

    _sink, review_node = make_human_review_entry()
    state = _review_state()
    state["risk_analysis"] = RiskAnalysisOutput(
        findings=[
            RiskFinding(
                risk_id="R-1",
                clause_id="canonical:doc-a:v1:chunk:1",
                label=RiskLabel.HIGH,
                title="Uncapped indemnity",
                explanation="no cap",
                citations=[Citation(claim="c", source="evh-1", confidence=0.9)],
                contract_index=0,
                support_status=ClaimSupportStatus.UNSUPPORTED,
            )
        ],
        summary="one unsupported finding",
    )
    state["compliance_result"] = ComplianceOutput(
        findings=[
            ComplianceFinding(
                finding_id="F-1",
                clause_id="canonical:doc-a:v1:chunk:2",
                statute="Act s1",
                is_compliant=False,
                explanation="contradicted by authority",
                citations=[Citation(claim="c", source="evh-2", confidence=0.9)],
                contract_index=1,
                support_status=ClaimSupportStatus.NOT_ASSESSABLE,
            )
        ],
        jurisdiction="India",
        summary="one not-assessable finding",
        compliance_status=ComplianceStatus.NOT_ASSESSABLE,
    )
    state["claim_ledger"] = [
        ClaimLedgerEntry(
            claim_id="L-1",
            claim="indemnity is capped",
            evidence_ids=["evh-1"],
            contract_index=0,
            support_status=ClaimSupportStatus.CONTRADICTED,
        )
    ]

    builder = StateGraph(LegalAgentState)
    builder.add_node("review", review_node)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    paused = await graph.ainvoke(state, {"configurable": {"thread_id": "thread-render"}})

    payload = paused["__interrupt__"][0].value
    rendered = {item["finding_id"]: item for item in payload["unsupported_items"]}
    assert rendered["R-1"]["support_status"] == "unsupported"
    assert rendered["R-1"]["contract_index"] == 0
    assert rendered["F-1"]["support_status"] == "not_assessable"
    assert rendered["F-1"]["contract_index"] == 1
    assert payload["compliance_status"] == "not_assessable"
    assert payload["compliance_not_assessable_reasons"] == ["insufficient_coverage"]
    assert rendered["L-1"]["support_status"] == "contradicted"
    assert rendered["L-1"]["title"] == "indemnity is capped"


async def test_supported_items_do_not_render_as_unsupported() -> None:
    from app.features.agent_saul.domain.state import Citation, ClaimSupportStatus, RiskFinding

    _sink, review_node = make_human_review_entry()
    state = _review_state()
    state["risk_analysis"] = RiskAnalysisOutput(
        findings=[
            RiskFinding(
                risk_id="R-ok",
                clause_id="canonical:doc-a:v1:chunk:1",
                label=RiskLabel.LOW,
                title="Supported",
                explanation="fine",
                citations=[Citation(claim="c", source="evh-1", confidence=0.9)],
                support_status=ClaimSupportStatus.SUPPORTED,
            )
        ],
        summary="supported",
    )

    builder = StateGraph(LegalAgentState)
    builder.add_node("review", review_node)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    paused = await graph.ainvoke(state, {"configurable": {"thread_id": "thread-clean"}})

    assert paused["__interrupt__"][0].value["unsupported_items"] == []


async def test_override_records_reason_identity_and_contract_identity() -> None:
    """The override reason and stamped contract identity survive into the report."""
    _sink, review_node = make_human_review_entry()
    state = _review_state()

    builder = StateGraph(LegalAgentState)
    builder.add_node("review", review_node)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-override"}}

    await graph.ainvoke(state, config)
    resumed = await graph.ainvoke(
        Command(
            resume={
                "action": "approve",
                "reviewer_id": "reviewer-9",
                "reviewer_role": "reviewer",
                "overrides": [
                    {
                        "clause_id": "canonical:doc-a:v1:chunk:2",
                        "original_label": "high",
                        "override_label": "medium",
                        "reason_code": "REVIEWER_JUDGEMENT",
                        "comment": "re-classified after authority re-read",
                    }
                ],
            }
        ),
        config,
    )

    review = resumed["human_review"]
    assert review.approved is True
    assert review.reviewer_id == "reviewer-9"
    override = review.overrides[0]
    assert override.reason_code == "REVIEWER_JUDGEMENT"
    # Contract identity is stamped from the checkpointed segment, never taken
    # from the resume payload.
    assert override.contract_index == 1
    assert resumed["self_reviewed"] is False


async def test_reviewer_supplied_contract_index_never_wins() -> None:
    """A resume payload naming a different contract than the checkpointed
    evidence cannot mislabel the override: the checkpoint identity always
    replaces the reviewer-supplied contract_index, so the identity that
    survives into FinalReport.human_overrides is the contract the override
    actually re-classifies."""
    _sink, review_node = make_human_review_entry()
    state = _review_state()

    builder = StateGraph(LegalAgentState)
    builder.add_node("review", review_node)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-override-mismatch"}}

    await graph.ainvoke(state, config)
    resumed = await graph.ainvoke(
        Command(
            resume={
                "action": "approve",
                "reviewer_id": "reviewer-9",
                "reviewer_role": "reviewer",
                "overrides": [
                    {
                        # The checkpointed segment for this clause is contract 1;
                        # the resume payload claims contract 0.
                        "clause_id": "canonical:doc-a:v1:chunk:2",
                        "original_label": "high",
                        "override_label": "medium",
                        "reason_code": "REVIEWER_JUDGEMENT",
                        "contract_index": 0,
                    }
                ],
            }
        ),
        config,
    )

    override = resumed["human_review"].overrides[0]
    assert override.contract_index == 1


async def test_contract_index_for_an_unattributed_clause_stays_none() -> None:
    """The reviewer never supplies the contract identity: a clause the
    checkpoint cannot attribute keeps ``contract_index=None`` even when the
    resume payload names one."""
    _sink, review_node = make_human_review_entry()
    state = _review_state()

    builder = StateGraph(LegalAgentState)
    builder.add_node("review", review_node)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-override-unknown"}}

    await graph.ainvoke(state, config)
    resumed = await graph.ainvoke(
        Command(
            resume={
                "action": "approve",
                "reviewer_id": "reviewer-9",
                "reviewer_role": "reviewer",
                "overrides": [
                    {
                        "clause_id": "canonical:doc-z:v1:chunk:99",
                        "original_label": "high",
                        "override_label": "medium",
                        "reason_code": "REVIEWER_JUDGEMENT",
                        "contract_index": 3,
                    }
                ],
            }
        ),
        config,
    )

    override = resumed["human_review"].overrides[0]
    assert override.contract_index is None
