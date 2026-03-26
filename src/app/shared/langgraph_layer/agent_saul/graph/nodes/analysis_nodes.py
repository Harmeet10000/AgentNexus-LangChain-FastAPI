"""
Analysis + terminal nodes:
  risk_analysis, compliance, grounding_verification,
  human_review (HITL), finalization, persist_memory.

Risk + Compliance run in PARALLEL via dual edges from relationship_mapping.
They join at grounding_verification which waits for both.

human_review is MANDATORY — legal liability requires a human signature.
persist_memory writes to Cognee via the injected async client.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.types import interrupt

from app.utils import logger

from ..state import (
    AgentError,
    ComplianceOutput,
    FinalReport,
    GroundingVerificationOutput,
    HITLInterruptType,
    HumanReviewOutput,
    LegalAgentState,
    ReviewOverride,
    RiskAnalysisOutput,
    RiskLabel,
    WorkflowStatus,
)

# ===========================================================================
# Risk Analysis Node  (create_react_agent — needs retrieval tools)
# ===========================================================================

_RISK_ANALYSIS_SYSTEM_PROMPT = """You are a senior legal risk analyst.

Perform multi-hop reasoning to identify contractual risks.

For each risk:
- Assign a risk label: low | medium | high | critical
- Explain the risk in plain English
- Cite SPECIFIC clauses, statutes, or precedents
- Suggest a revision if applicable

Special focus for Indian law:
- Unlimited liability clauses
- One-sided termination rights
- Weak arbitration seats
- Non-enforceable conditions

Citation enforcement: EVERY risk finding MUST include citations.
Guardrail: If you cannot cite a source, do not make the claim.
"""


def make_risk_analysis_node(
    risk_agent: Any,  # create_react_agent CompiledStateGraph
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def risk_analysis_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="risk_analysis",
            clause_count=len(state["segments"]),
            entity_count=len(state["extracted_entities"]),
        )

        context = _build_analysis_context(state)
        result = await risk_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=_RISK_ANALYSIS_SYSTEM_PROMPT),
                    HumanMessage(content=context),
                    *state["messages"],
                ]
            }
        )

        # Extract structured output from agent messages
        risk_output = _extract_risk_output(result["messages"])
        log.info(
            "risk_analysis_completed",
            finding_count=len(risk_output.findings),
            overall_label=risk_output.overall_label,
        )

        return {
            "risk_analysis": risk_output,
            "status": WorkflowStatus.CHECKING_COMPLIANCE,
        }

    return risk_analysis_node


def _build_analysis_context(state: LegalAgentState) -> str:
    clauses = "\n".join(
        f"[{seg.clause_type}] {seg.clause_id}: {seg.text[:300]}"
        for seg in state["segments"]
    )
    entities = "\n".join(
        f"{e.entity_type}: {e.value} (party: {e.party or 'N/A'})"
        for e in state["extracted_entities"]
    )
    relationships = "\n".join(
        f"{r.from_node} → {r.relationship} → {r.to_node} (clause: {r.clause_id})"
        for r in state["relationships"]
    )
    return f"CLAUSES:\n{clauses}\n\nENTITIES:\n{entities}\n\nRELATIONSHIPS:\n{relationships}"


def _extract_risk_output(_messages: list[Any]) -> RiskAnalysisOutput:
    # TODO: parse structured output from last AI message using with_structured_output
    # For now return a stub that type-checks
    return RiskAnalysisOutput(
        findings=[],
        overall_label=RiskLabel.LOW,
        summary="Risk analysis pending tool integration",
    )


# ===========================================================================
# Compliance Node  (create_react_agent — retrieval-first, no hallucinations)
# ===========================================================================

_COMPLIANCE_SYSTEM_PROMPT = """You are a legal compliance analyst specialising in Indian law.

Tasks:
1. Check statute applicability (IT Act, Contract Act, GDPR equivalents, SEBI, etc.)
2. Surface binding precedents from Indian courts
3. Detect cross-jurisdictional conflicts

STRICT rule: If retrieved sources < confidence threshold → respond:
  "Insufficient legal basis — cannot make compliance determination for [clause_id]"

DO NOT hallucinate statutes, section numbers, or case citations.
Citation enforcement: EVERY finding MUST include citations.
"""


def make_compliance_node(
    compliance_agent: Any,  # create_react_agent CompiledStateGraph
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def compliance_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="compliance")

        context = _build_analysis_context(state)
        result = await compliance_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=_COMPLIANCE_SYSTEM_PROMPT),
                    HumanMessage(content=context),
                    *state["messages"],
                ]
            }
        )

        compliance_output = _extract_compliance_output(result["messages"])
        log.info(
            "compliance_completed",
            finding_count=len(compliance_output.findings),
            overall_compliant=compliance_output.overall_compliant,
        )

        return {
            "compliance_result": compliance_output,
            "status": WorkflowStatus.VERIFYING_GROUNDING,
        }

    return compliance_node


def _extract_compliance_output(_messages: list[Any]) -> ComplianceOutput:
    # TODO: parse structured output from last AI message
    return ComplianceOutput(
        findings=[],
        jurisdiction="India",
        overall_compliant=True,
        summary="Compliance check pending tool integration",
    )


# ===========================================================================
# Grounding Verification Node (join after parallel risk + compliance)
# ===========================================================================

_GROUNDING_SYSTEM_PROMPT = """You are a grounding verifier.

Review all risk and compliance findings.
Identify any claims that lack sufficient citation support.
Flag unverified claims that should not be presented to the user.

Output ONLY GroundingVerificationOutput schema.
"""


def make_grounding_verification_node(
    grounding_llm: Runnable[list[Any], GroundingVerificationOutput],
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def grounding_verification_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="grounding_verification")

        summary_parts: list[str] = []
        risk_analysis = state.get("risk_analysis")
        if risk_analysis:
            summary_parts.append(f"RISK SUMMARY: {risk_analysis.summary}")
        compliance_result = state.get("compliance_result")
        if compliance_result:
            summary_parts.append(f"COMPLIANCE SUMMARY: {compliance_result.summary}")

        messages = [
            SystemMessage(content=_GROUNDING_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(summary_parts)),
        ]
        result: GroundingVerificationOutput = await grounding_llm.ainvoke(messages)
        log.info("grounding_verified", verified=result.verified)

        return {
            "grounding": result,
            "status": WorkflowStatus.AWAITING_HUMAN_REVIEW,
        }

    return grounding_verification_node


# ===========================================================================
# Human Review Node (MANDATORY HITL)
# ===========================================================================


def make_human_review_node() -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def human_review_node(state: LegalAgentState) -> dict[str, Any]:
        risk_analysis = state.get("risk_analysis")
        compliance_result = state.get("compliance_result")
        grounding = state.get("grounding")

        log = logger.bind(
            node="human_review",
            risk_findings=len(risk_analysis.findings if risk_analysis else []),
        )

        # Build review payload for the human interface
        review_payload: dict[str, Any] = {
            "type": HITLInterruptType.HUMAN_REVIEW_REQUIRED,
            "risk_summary": risk_analysis.summary if risk_analysis else None,
            "compliance_summary": compliance_result.summary if compliance_result else None,
            "unverified_claims": grounding.unverified_claims if grounding else [],
            "segments": [seg.model_dump() for seg in state["segments"][:20]],  # cap payload size
            "message": "Please review findings, add overrides if needed, and approve to finalize",
        }

        # MANDATORY interrupt — no bypass
        human_response: dict[str, Any] = interrupt(review_payload)

        if human_response.get("action") == "reject":
            log.warning("human_review_rejected")
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="human_review",
                        code="REVIEW_REJECTED",
                        message=human_response.get("feedback") or "Rejected at human review",
                        retryable=True,
                    )
                ],
            }

        raw_overrides = human_response.get("overrides") or []
        overrides = [ReviewOverride.model_validate(o) for o in raw_overrides]

        review_output = HumanReviewOutput(
            reviewer_id=human_response.get("reviewer_id", "unknown"),
            reviewer_role=human_response.get("reviewer_role", "reviewer"),
            overrides=overrides,
            approved=human_response.get("action") == "approve",
            notes=human_response.get("feedback"),
        )

        log.info(
            "human_review_completed",
            override_count=len(overrides),
            approved=review_output.approved,
        )

        return {
            "human_review": review_output,
            "status": WorkflowStatus.FINALIZING,
        }

    return human_review_node


# ===========================================================================
# Finalization Node (MANDATORY — structured final report)
# ===========================================================================

_FINALIZATION_SYSTEM_PROMPT = """You are the legal report finalizer for Agent Saul.

Synthesize all analysis into a final report for the user.

Include:
- Executive summary (plain English)
- All risk findings (with human overrides applied)
- All compliance findings
- Suggested actions the user should take
- All citations used

Citation enforcement: output MUST include every citation used in findings.
Output ONLY FinalReport schema.
"""


def make_finalization_node(
    finalization_llm: Runnable[list[Any], FinalReport],
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    async def finalization_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="finalization", doc_id=state["doc_id"])
        risk_analysis = state.get("risk_analysis")
        compliance_result = state.get("compliance_result")
        human_review = state.get("human_review")

        context_parts: list[str] = [
            f"Document ID: {state['doc_id']}",
            f"User query: {state['user_query']}",
        ]
        if risk_analysis:
            context_parts.append(f"Risk summary: {risk_analysis.summary}")
        if compliance_result:
            context_parts.append(f"Compliance: {compliance_result.summary}")
        if human_review and human_review.overrides:
            overrides_text = "\n".join(
                f"Override: {o.clause_id} {o.original_label} → {o.override_label}: {o.reason_code}"
                for o in human_review.overrides
            )
            context_parts.append(f"Human overrides:\n{overrides_text}")

        messages = [
            SystemMessage(content=_FINALIZATION_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(context_parts)),
        ]
        report: FinalReport = await finalization_llm.ainvoke(messages)
        log.info("finalization_completed", suggested_actions=len(report.suggested_actions))

        return {
            "final_report": report,
            "status": WorkflowStatus.PERSISTING_MEMORY,
        }

    return finalization_node


# ===========================================================================
# PersistMemory Node (MANDATORY — Cognee long-term memory)
# ===========================================================================


def make_persist_memory_node(
    _cognee_client: Any,  # injected Cognee async client
) -> Callable[[LegalAgentState], Awaitable[dict[str, Any]]]:
    """
    Writes final report + relationships + entities to Cognee.

    Cognee namespace: ["user_id", "legal_domain"]
    The cognee_client is injected from app.state at graph build time.

    TODO: wire actual Cognee client methods:
        await cognee_client.add(data, dataset_name=namespace)
        await cognee_client.cognify()
    """

    async def persist_memory_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="persist_memory",
            user_id=state["user_id"],
            doc_id=state["doc_id"],
        )

        namespace = f"{state['user_id']}.legal"
        long_term_refs: list[str] = list(state.get("long_term_refs", []))

        try:
            if state.get("final_report"):
                # TODO: replace with actual Cognee write
                # await cognee_client.add(
                #     state["final_report"].model_dump_json(),
                #     dataset_name=namespace,
                # )
                ref_key = f"{namespace}.{state['doc_id']}.report"
                long_term_refs.append(ref_key)

            if state.get("relationships"):
                # TODO: persist relationship graph to Cognee
                # await cognee_client.add(relationships_payload, dataset_name=namespace)
                rel_key = f"{namespace}.{state['doc_id']}.relationships"
                long_term_refs.append(rel_key)

            # TODO: await cognee_client.cognify()  # triggers Cognee's knowledge graph build
            log.info("persist_memory_completed", ref_count=len(long_term_refs))

        except Exception as exc:
            log.exception("persist_memory_failed", error=str(exc))
            return {
                "long_term_refs": long_term_refs,
                "status": WorkflowStatus.COMPLETED,
                "errors": [
                    AgentError(
                        node="persist_memory",
                        code="COGNEE_WRITE_FAILED",
                        message=str(exc),
                        retryable=True,
                    )
                ],
            }

        return {
            "long_term_refs": long_term_refs,
            "status": WorkflowStatus.COMPLETED,
        }

    return persist_memory_node
