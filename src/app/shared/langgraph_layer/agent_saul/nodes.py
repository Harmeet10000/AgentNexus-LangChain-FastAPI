"""Agent Saul node implementations and routing helpers."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

from app.utils import logger

from .prompts import (
    _COMPLIANCE_SYSTEM_PROMPT,
    _ENTITY_EXTRACTION_SYSTEM_PROMPT,
    _FINALIZATION_SYSTEM_PROMPT,
    _GROUNDING_SYSTEM_PROMPT,
    _NORMALIZATION_SYSTEM_PROMPT,
    _ORCHESTRATOR_SYSTEM_PROMPT,
    _PLANNER_SYSTEM_PROMPT,
    _QNA_SYSTEM_PROMPT,
    _RELATIONSHIP_MAPPING_SYSTEM_PROMPT,
    _RISK_ANALYSIS_SYSTEM_PROMPT,
    _SEGMENTATION_SYSTEM_PROMPT,
)
from .state import (
    AgentError,
    CitedEntity,
    ClauseExtractionInput,
    ClauseSegment,
    ComplianceOutput,
    FinalReport,
    GroundingVerificationOutput,
    HITLInterruptType,
    HumanReviewOutput,
    LegalAgentState,
    LegalRelationship,
    NormalizedDocument,
    OrchestratorAction,
    OrchestratorActionType,
    PlanStep,
    ReviewOverride,
    RiskAnalysisOutput,
    RiskLabel,
    WorkflowStatus,
)

_CLARIFICATION_THRESHOLD = 0.72
_OCR_CONFIDENCE_THRESHOLD = 0.85
_MAX_RETRIES = 3

_VALID_WORKER_NODES = frozenset(
    {
        "ingestion",
        "normalization",
        "segmentation",
        "entity_extraction",
        "relationship_mapping",
        "risk_analysis",
        "compliance",
        "grounding_verification",
        "finalization",
    }
)


class QnAOutput(BaseModel):
    """Structured output schema for the QnA LLM call."""

    intent: str = Field(description="Restated user intent as a clear actionable objective")
    confidence: float = Field(ge=0.0, le=1.0)
    clarification_question: str | None = Field(
        default=None,
        description="Single clarifying question when confidence < threshold",
    )
    document_type_guess: Literal["NDA", "MSA", "SLA", "employment", "lease", "other"] | None = None


class PlannerOutput(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=10)
    rationale: str = Field(description="Why this plan was chosen")


class ClauseSegmentationOutput(BaseModel):
    segments: list[ClauseSegment] = Field(min_length=1)


class EntityExtractionOutput(BaseModel):
    entities: list[CitedEntity]
    overall_confidence: float = Field(ge=0.0, le=1.0)


class RelationshipMappingOutput(BaseModel):
    relationships: list[LegalRelationship]


type StateNode = Callable[[LegalAgentState], Awaitable[dict[str, Any]]]


def make_gateway_node() -> StateNode:
    async def gateway_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            correlation_id=state["correlation_id"],
        )

        if not state.get("doc_id"):
            log.error("gateway_missing_doc_id")
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="gateway",
                        code="MISSING_DOC_ID",
                        message="doc_id is required to start the pipeline",
                        retryable=False,
                    )
                ],
            }

        working_memory: dict[str, Any] = dict(state.get("working_memory", {}))
        working_memory["gateway_validated"] = True
        working_memory["session_start_ts"] = _utc_now_iso()

        log.info("gateway_validated")

        return {
            "working_memory": working_memory,
            "status": WorkflowStatus.QNA_CLARIFICATION,
        }

    return gateway_node


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def make_qna_node(qna_llm: Runnable[list[Any], QnAOutput]) -> StateNode:
    async def qna_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="qna",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
        )

        messages = [SystemMessage(content=_QNA_SYSTEM_PROMPT), *state["messages"]]
        result: QnAOutput = await qna_llm.ainvoke(messages)

        log.info("qna_scored", confidence=result.confidence, intent=result.intent)

        if result.confidence < _CLARIFICATION_THRESHOLD:
            user_answer: dict[str, Any] = interrupt(
                {
                    "type": HITLInterruptType.CLARIFICATION_NEEDED,
                    "question": result.clarification_question or "Could you clarify your query?",
                    "current_confidence": result.confidence,
                    "message": "Query requires clarification before proceeding",
                }
            )
            answer_text: str = user_answer.get("feedback") or ""
            log.info("qna_clarification_received", answer_length=len(answer_text))

            return {
                "messages": [HumanMessage(content=answer_text)],
                "qna_confidence": result.confidence,
                "status": WorkflowStatus.QNA_CLARIFICATION,
            }

        working_memory = dict(state.get("working_memory", {}))
        working_memory["clarified_intent"] = result.intent
        if result.document_type_guess:
            working_memory["document_type"] = result.document_type_guess

        return {
            "qna_confidence": result.confidence,
            "working_memory": working_memory,
            "status": WorkflowStatus.PLAN_PENDING,
        }

    return qna_node


def route_after_qna(state: LegalAgentState) -> str:
    if state["status"] == WorkflowStatus.QNA_CLARIFICATION:
        return "qna"
    return "orchestrator"


def make_orchestrator_node(
    orchestrator_llm: Runnable[list[Any], OrchestratorAction],
) -> StateNode:
    async def orchestrator_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="orchestrator",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            current_step=state["current_step"],
        )

        messages = [
            SystemMessage(content=_ORCHESTRATOR_SYSTEM_PROMPT),
            *state["messages"],
        ]
        action: OrchestratorAction = await orchestrator_llm.ainvoke(messages)

        if (
            action.action_type == OrchestratorActionType.CONTINUE
            and action.target_node not in _VALID_WORKER_NODES
        ):
            log.error("orchestrator_invalid_target", target=action.target_node)
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="orchestrator",
                        code="INVALID_TARGET_NODE",
                        message=f"Orchestrator routed to unknown node: {action.target_node}",
                        retryable=False,
                    )
                ],
            }

        log.info(
            "orchestrator_action_decided",
            action_type=action.action_type,
            target=action.target_node,
            reflection=action.reflection[:120],
        )

        return {
            "orchestrator_action": action,
            "current_step": state["current_step"] + 1,
        }

    return orchestrator_node


def route_from_orchestrator(state: LegalAgentState) -> str:
    action = state.get("orchestrator_action")
    if action is None:
        return "planner"

    if state.get("status") == WorkflowStatus.PLAN_REJECTED:
        return "planner"

    match action.action_type:
        case OrchestratorActionType.START_PIPELINE:
            return "ingestion"
        case OrchestratorActionType.CONTINUE:
            return action.target_node or "ingestion"
        case OrchestratorActionType.SYNTHESIZE:
            return "finalization"
        case OrchestratorActionType.DONE:
            return END
        case _:
            return "planner"


def make_planner_node(planner_llm: Runnable[list[Any], PlannerOutput]) -> StateNode:
    async def planner_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="planner",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
        )

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            *state["messages"],
        ]
        result: PlannerOutput = await planner_llm.ainvoke(messages)
        log.info("planner_plan_generated", step_count=len(result.steps))

        human_response: dict[str, Any] = interrupt(
            {
                "type": HITLInterruptType.PLAN_APPROVAL,
                "plan": [step.model_dump() for step in result.steps],
                "rationale": result.rationale,
                "message": "Please review and approve the execution plan",
            }
        )

        action: str = human_response.get("action", "approve")

        if action == "reject":
            log.info("planner_plan_rejected")
            return {
                "status": WorkflowStatus.PLAN_REJECTED,
                "errors": [
                    AgentError(
                        node="planner",
                        code="PLAN_REJECTED",
                        message=human_response.get("feedback") or "Plan rejected by reviewer",
                        retryable=True,
                    )
                ],
            }

        if action == "modify":
            raw_steps: list[dict[str, Any]] = human_response.get("modified_plan") or []
            approved_plan = [PlanStep.model_validate(s) for s in raw_steps] or result.steps
            log.info("planner_plan_modified", step_count=len(approved_plan))
        else:
            approved_plan = result.steps

        return {
            "plan": list(approved_plan),
            "current_step": 0,
            "plan_approved": True,
            "status": WorkflowStatus.PLAN_APPROVED,
        }

    return planner_node


def make_ingestion_node() -> StateNode:
    async def ingestion_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="ingestion",
            doc_id=state["doc_id"],
            user_id=state["user_id"],
        )
        log.info("ingestion_started")

        retry_count = state.get("retry_count", 0)
        if retry_count >= _MAX_RETRIES:
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="ingestion",
                        code="MAX_RETRIES_EXCEEDED",
                        message=f"Ingestion failed after {_MAX_RETRIES} attempts",
                        retryable=False,
                    )
                ],
            }

        text: str = ""
        confidence: float = 1.0

        if confidence < _OCR_CONFIDENCE_THRESHOLD:
            log.warning("ingestion_low_ocr_confidence", confidence=confidence)
            human_response: dict[str, Any] = interrupt(
                {
                    "type": HITLInterruptType.OCR_REUPLOAD,
                    "confidence": confidence,
                    "message": (
                        f"Document OCR confidence ({confidence:.0%}) is below threshold. "
                        "Please re-upload a higher-quality scan."
                    ),
                }
            )
            new_doc_id: str | None = human_response.get("new_doc_id")
            if new_doc_id:
                return {
                    "doc_id": new_doc_id,
                    "retry_count": retry_count + 1,
                    "status": WorkflowStatus.INGESTING,
                }
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="ingestion",
                        code="LOW_OCR_CONFIDENCE",
                        message=f"OCR confidence {confidence:.0%} - user declined re-upload",
                        retryable=False,
                    )
                ],
            }

        log.info("ingestion_completed", text_length=len(text), confidence=confidence)
        return {
            "document_text": text,
            "status": WorkflowStatus.NORMALIZING,
        }

    return ingestion_node


def make_normalization_node(
    normalization_llm: Runnable[list[Any], NormalizedDocument],
) -> StateNode:
    async def normalization_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="normalization", doc_id=state["doc_id"])

        if not state.get("document_text"):
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="normalization",
                        code="MISSING_DOCUMENT_TEXT",
                        message="document_text not populated by ingestion",
                        retryable=False,
                    )
                ],
            }

        messages = [
            SystemMessage(content=_NORMALIZATION_SYSTEM_PROMPT),
            HumanMessage(content=state["document_text"]),
        ]
        result: NormalizedDocument = await normalization_llm.ainvoke(messages)
        log.info("normalization_completed", section_count=len(result.sections))

        return {
            "normalized_document": result,
            "status": WorkflowStatus.SEGMENTING,
        }

    return normalization_node


def make_segmentation_node(
    segmentation_llm: Runnable[list[Any], ClauseSegmentationOutput],
) -> StateNode:
    async def segmentation_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="segmentation", doc_id=state["doc_id"])

        if not state.get("normalized_document"):
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="segmentation",
                        code="MISSING_NORMALIZED_DOCUMENT",
                        message="normalized_document not populated by normalization node",
                        retryable=False,
                    )
                ],
            }

        normalized_document = state["normalized_document"]
        if normalized_document is None:
            return {
                "status": WorkflowStatus.FAILED,
                "errors": [
                    AgentError(
                        node="segmentation",
                        code="MISSING_NORMALIZED_DOCUMENT",
                        message="normalized_document not populated by normalization node",
                        retryable=False,
                    )
                ],
            }

        doc_text = "\n".join(s.content for s in normalized_document.sections)
        messages = [
            SystemMessage(content=_SEGMENTATION_SYSTEM_PROMPT),
            HumanMessage(content=doc_text),
        ]
        result: ClauseSegmentationOutput = await segmentation_llm.ainvoke(messages)
        log.info("segmentation_completed", segment_count=len(result.segments))

        return {
            "segments": list(result.segments),
            "status": WorkflowStatus.EXTRACTING_ENTITIES,
        }

    return segmentation_node


def dispatch_entity_extraction(state: LegalAgentState) -> list[Send]:
    document_context: dict[str, Any] = {
        "jurisdiction": state["working_memory"].get("jurisdiction", "India"),
        "document_type": state["working_memory"].get("document_type", "unknown"),
        "doc_id": state["doc_id"],
    }
    return [
        Send(
            "entity_extraction",
            ClauseExtractionInput(
                clause=segment,
                document_context=document_context,
            ).model_dump(),
        )
        for segment in state["segments"]
    ]


def make_entity_extraction_node(
    entity_llm: Runnable[list[Any], EntityExtractionOutput],
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    async def entity_extraction_node(state: dict[str, Any]) -> dict[str, Any]:
        clause = ClauseSegment.model_validate(state["clause"])
        doc_context: dict[str, Any] = state.get("document_context", {})

        log = logger.bind(node="entity_extraction", clause_id=clause.clause_id)

        prompt = (
            f"Document type: {doc_context.get('document_type', 'unknown')}\n"
            f"Jurisdiction: {doc_context.get('jurisdiction', 'India')}\n"
            f"Clause ID: {clause.clause_id}\n"
            f"Clause type: {clause.clause_type}\n\n"
            f"Clause text:\n{clause.text}"
        )
        messages = [
            SystemMessage(content=_ENTITY_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        result: EntityExtractionOutput = await entity_llm.ainvoke(messages)
        log.info("entity_extraction_done", entity_count=len(result.entities))

        return {"extracted_entities": list(result.entities)}

    return entity_extraction_node


def make_relationship_mapping_node(
    relationship_llm: Runnable[list[Any], RelationshipMappingOutput],
) -> StateNode:
    async def relationship_mapping_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="relationship_mapping",
            entity_count=len(state["extracted_entities"]),
        )

        entity_summary = "\n".join(
            f"[{e.entity_type}] {e.value} (clause: {e.clause_id}, party: {e.party or 'N/A'})"
            for e in state["extracted_entities"]
        )
        messages = [
            SystemMessage(content=_RELATIONSHIP_MAPPING_SYSTEM_PROMPT),
            HumanMessage(content=entity_summary),
        ]
        result: RelationshipMappingOutput = await relationship_llm.ainvoke(messages)
        log.info("relationship_mapping_done", relationship_count=len(result.relationships))

        return {
            "relationships": list(result.relationships),
            "status": WorkflowStatus.ANALYZING_RISKS,
        }

    return relationship_mapping_node


def make_risk_analysis_node(risk_agent: Any) -> StateNode:
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
        f"{r.from_node} -> {r.relationship} -> {r.to_node} (clause: {r.clause_id})"
        for r in state["relationships"]
    )
    return f"CLAUSES:\n{clauses}\n\nENTITIES:\n{entities}\n\nRELATIONSHIPS:\n{relationships}"


def _extract_risk_output(_messages: list[Any]) -> RiskAnalysisOutput:
    return RiskAnalysisOutput(
        findings=[],
        overall_label=RiskLabel.LOW,
        summary="Risk analysis pending tool integration",
    )


def make_compliance_node(compliance_agent: Any) -> StateNode:
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
    return ComplianceOutput(
        findings=[],
        jurisdiction="India",
        overall_compliant=True,
        summary="Compliance check pending tool integration",
    )


def make_grounding_verification_node(
    grounding_llm: Runnable[list[Any], GroundingVerificationOutput],
) -> StateNode:
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


def make_human_review_node() -> StateNode:
    async def human_review_node(state: LegalAgentState) -> dict[str, Any]:
        risk_analysis = state.get("risk_analysis")
        compliance_result = state.get("compliance_result")
        grounding = state.get("grounding")

        log = logger.bind(
            node="human_review",
            risk_findings=len(risk_analysis.findings if risk_analysis else []),
        )

        review_payload: dict[str, Any] = {
            "type": HITLInterruptType.HUMAN_REVIEW_REQUIRED,
            "risk_summary": risk_analysis.summary if risk_analysis else None,
            "compliance_summary": compliance_result.summary if compliance_result else None,
            "unverified_claims": grounding.unverified_claims if grounding else [],
            "segments": [seg.model_dump() for seg in state["segments"][:20]],
            "message": "Please review findings, add overrides if needed, and approve to finalize",
        }

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


def make_finalization_node(
    finalization_llm: Runnable[list[Any], FinalReport],
) -> StateNode:
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
                f"Override: {o.clause_id} {o.original_label} -> {o.override_label}: {o.reason_code}"
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


def make_persist_memory_node(_cognee_client: Any) -> StateNode:
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
                ref_key = f"{namespace}.{state['doc_id']}.report"
                long_term_refs.append(ref_key)

            if state.get("relationships"):
                rel_key = f"{namespace}.{state['doc_id']}.relationships"
                long_term_refs.append(rel_key)

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
