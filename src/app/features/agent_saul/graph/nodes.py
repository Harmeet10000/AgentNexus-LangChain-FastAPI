"""Agent Saul node implementations and routing helpers."""

import asyncio
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, assert_never, cast

from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field, ValidationError
from returns.result import Failure, Success

from app.features.agent_saul.adapters.document_source import (
    CanonicalDocumentChunk,
    DocumentTextFailure,
    DocumentTextFailureCode,
    document_content_source_from_config,
)
from app.features.agent_saul.domain.budget import (
    BudgetDimension,
    BudgetExhaustedError,
    BudgetLedger,
)
from app.features.agent_saul.domain.state import (
    GRAPH_NODE_NAMES,
    AgentError,
    AssessmentStatus,
    Citation,
    CitedEntity,
    ClaimLedgerEntry,
    ClaimSupportStatus,
    ClauseSegment,
    ClauseType,
    ComplianceOutput,
    ComplianceStatus,
    EvidenceRef,
    FinalReport,
    GroundingStatus,
    GroundingVerificationOutput,
    HITLInterruptType,
    HumanReviewOutput,
    IngestionDocumentState,
    IngestionDocumentStatus,
    LegalAgentState,
    LegalRelationship,
    NormalizedDocument,
    OrchestratorAction,
    OrchestratorActionType,
    PlanActionType,
    PlanStep,
    ReviewOverride,
    RiskAnalysisOutput,
    WorkflowStatus,
    hydrate_state,
)
from app.features.documents.api.http.dto import EvidenceSearchRequest, EvidenceSearchResponse
from app.features.documents.application.tools.authorization import ToolRuntimeContext
from app.observability.logging import logger
from app.shared.langchain_layer.messages import make_delegation_pair

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

if TYPE_CHECKING:
    from app.features.documents.api.http.dto import EvidencePolicy

_CLARIFICATION_THRESHOLD = 0.72
_OCR_CONFIDENCE_THRESHOLD = 0.85
_MAX_RETRIES = 3
_HUMAN_REVIEW_SEGMENT_PREVIEW = 20
_REFLECTION_LOG_CHARS = 120
_CLAUSE_CONTEXT_CHARS = 300


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


# Subset of GRAPH_NODE_NAMES reachable directly from the orchestrator's CONTINUE action.
_VALID_WORKER_NODES = GRAPH_NODE_NAMES - frozenset(
    {
        "gateway",
        "qna",
        "orchestrator",
        "planner",
        "planner_generate",
        "persist_memory",
        "human_review",
    }
)


def _fail(
    node: str,
    code: str,
    message: str,
    *,
    status: WorkflowStatus = WorkflowStatus.FAILED,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "errors": [AgentError(node=node, code=code, message=message, retryable=retryable)],
    }


def _budget_gate(
    state: LegalAgentState,
    dimension: BudgetDimension,
    amount: int = 1,
) -> tuple[BudgetLedger, dict[str, Any] | None]:
    """Consume persisted usage before an operation; exhaustion is terminal."""
    raw = state.get("budget_ledger")
    if isinstance(raw, BudgetLedger):
        ledger = raw
    elif isinstance(raw, dict):
        try:
            ledger = BudgetLedger.model_validate(raw)
        except ValidationError:
            ledger = BudgetLedger(exhausted=True)
    else:
        ledger = BudgetLedger()
    try:
        return ledger.consume(dimension, amount), None
    except BudgetExhaustedError as exc:
        exhausted = ledger.model_copy(
            update={"exhausted": True, "exhausted_dimension": exc.dimension}
        )
        return exhausted, _fail(
            "budget",
            "BUDGET_EXHAUSTED",
            str(exc),
            retryable=False,
        )


async def _invoke_structured(
    llm: Runnable[list[Any], Any],
    system_prompt: str,
    human_content: str | None,
) -> Any:
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    return await llm.ainvoke(messages)


def _agent_messages(result: Any) -> list[Any]:
    """Extract the public message trace from a create_agent result."""
    if not isinstance(result, dict):
        return []
    messages = result.get("messages", [])
    return messages if isinstance(messages, list) else []


def _consume_observed_tool_calls(
    ledger: BudgetLedger, result: Any
) -> tuple[BudgetLedger, dict[str, Any] | None]:
    """Charge tool calls emitted by a nested agent before accepting its output."""
    tool_calls = sum(
        len(getattr(message, "tool_calls", None) or []) for message in _agent_messages(result)
    )
    if not tool_calls:
        return ledger, None
    try:
        return ledger.consume(BudgetDimension.TOOL_CALLS, tool_calls), None
    except BudgetExhaustedError as exc:
        return ledger.model_copy(
            update={"exhausted": True, "exhausted_dimension": exc.dimension}
        ), _fail("budget", "BUDGET_EXHAUSTED", str(exc), retryable=False)


def _consume_observed_token_usage(
    ledger: BudgetLedger, result: Any
) -> tuple[BudgetLedger, dict[str, Any] | None]:
    """Charge provider usage metadata when a model exposes it."""
    usage: Any = getattr(result, "usage_metadata", None)
    if usage is None and isinstance(result, dict):
        usage = result.get("usage_metadata") or result.get("usage")
    if usage is None:
        metadata = getattr(result, "response_metadata", None)
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
    if not isinstance(usage, Mapping):
        return ledger, None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return ledger, None
    try:
        charged = ledger.consume(
            BudgetDimension.TOKEN_USAGE,
            amount=0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except (BudgetExhaustedError, ValueError) as exc:
        dimension = (
            exc.dimension if isinstance(exc, BudgetExhaustedError) else BudgetDimension.TOKEN_USAGE
        )
        return ledger.model_copy(
            update={"exhausted": True, "exhausted_dimension": dimension}
        ), _fail("budget", "BUDGET_EXHAUSTED", str(exc), retryable=False)
    return charged, None


def _charge_model_result(
    ledger: BudgetLedger, result: Any
) -> tuple[BudgetLedger, dict[str, Any] | None]:
    """Apply post-call tool and provider-token accounting in one boundary."""
    ledger, blocked = _consume_observed_tool_calls(ledger, result)
    if blocked is not None:
        return ledger, blocked
    return _consume_observed_token_usage(ledger, result)


def _canonical_quote_digests(content: str) -> tuple[str, ...]:
    """Hash bounded contiguous word windows from a canonical dereference."""
    words = content.split()
    digests: set[str] = {hashlib.sha256(content.encode()).hexdigest()}
    max_window = min(len(words), 128)
    for start in range(len(words)):
        digests.update(
            hashlib.sha256(" ".join(words[start:end]).encode()).hexdigest()
            for end in range(start + 1, min(len(words), start + max_window) + 1)
        )
    return tuple(sorted(digests))


def _evidence_metadata_from_messages(
    messages: list[Any],
) -> tuple[dict[str, EvidenceRef], dict[str, int]]:
    """Extract only server-issued evidence metadata from tool messages.

    Nested specialist graphs expose tool responses as messages.  We retain
    handles, versions, and coverage, but deliberately discard snippets and
    read content before anything reaches the outer checkpoint.
    """

    registry: dict[str, EvidenceRef] = {}
    coverage: dict[str, int] = {}

    def visit(value: object) -> None:  # noqa: PLR0912 — recursively normalizes tool envelopes
        if isinstance(value, str):
            try:
                visit(json.loads(value))
            except (TypeError, ValueError):
                return
            return
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, Mapping):
            return
        raw_coverage = value.get("coverage")
        if isinstance(raw_coverage, Mapping):
            for key, item in raw_coverage.items():
                if isinstance(item, int):
                    coverage[str(key)] = item
        raw_evidence = value.get("evidence")
        if isinstance(raw_evidence, Mapping):
            try:
                ref = EvidenceRef.model_validate(raw_evidence)
            except ValidationError:
                ref = None
            if ref is not None:
                nested_content = value.get("content")
                if isinstance(nested_content, str):
                    ref = ref.model_copy(
                        update={
                            "quote_digest": hashlib.sha256(nested_content.encode()).hexdigest(),
                            "quote_digests": _canonical_quote_digests(nested_content),
                        }
                    )
                registry[ref.evidence_id] = ref
        # Read responses contain a full ``content`` field.  Only its digest is
        # retained so quote verification can happen without checkpointing raw
        # document text.
        evidence_id = value.get("evidence_id")
        content = value.get("content")
        if isinstance(evidence_id, str) and isinstance(content, str):
            ref = registry.get(evidence_id)
            if ref is not None:
                registry[evidence_id] = ref.model_copy(
                    update={
                        "quote_digest": hashlib.sha256(content.encode()).hexdigest(),
                        "quote_digests": _canonical_quote_digests(content),
                    }
                )
        raw_items = value.get("items")
        if isinstance(raw_items, list):
            for item in raw_items:
                visit(item)

    for message in messages:
        visit(getattr(message, "content", message))
    return registry, coverage


def _merge_evidence_metadata(
    state: LegalAgentState,
    result: Any,
) -> tuple[dict[str, EvidenceRef], dict[str, int]]:
    registry, coverage = _evidence_metadata_from_messages(_agent_messages(result))
    run_id = state.get("run_id") or state.get("correlation_id") or state.get("thread_id", "")
    thread_id = state.get("thread_id", "")
    evidence_checkpoint_id = state.get("evidence_checkpoint_id") or f"run:{run_id}"
    registry = {
        evidence_id: ref.model_copy(
            update={
                "run_id": ref.run_id or run_id,
                "thread_id": ref.thread_id or thread_id,
                "checkpoint_id": ref.checkpoint_id or evidence_checkpoint_id,
            }
        )
        for evidence_id, ref in registry.items()
    }
    merged_registry = dict(state.get("evidence_registry", {}))
    merged_registry.update(registry)
    merged_coverage = dict(state.get("coverage", {}))
    merged_coverage.update(coverage)
    return merged_registry, merged_coverage


def _checkpoint_scope(state: LegalAgentState) -> str:
    """Return the same concrete checkpoint marker used when issuing handles."""

    run_id = state.get("run_id") or state.get("correlation_id") or state.get("thread_id", "")
    return state.get("evidence_checkpoint_id") or f"run:{run_id}"


def _agent_structured_response(result: Any, schema: type[BaseModel]) -> BaseModel | None:
    """Read the response-format value emitted by ``create_agent``.

    LangChain's agent graph stores a typed response under
    ``structured_response``.  Older test doubles and provider failures often
    return only a message list; those are deliberately *not* interpreted as a
    legal finding.  A dict response is accepted only after Pydantic validates
    it against the node's schema.
    """
    value: Any = result
    if isinstance(result, dict):
        value = result.get("structured_response")
    if value is None:
        return None
    if isinstance(value, schema):
        return value
    if isinstance(value, dict):
        try:
            return schema.model_validate(value)
        except ValidationError:
            return None
    return None


def _analysis_failure(
    node: str,
    code: str,
    message: str,
) -> AgentError:
    return AgentError(node=node, code=code, message=message, retryable=True)


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

# Ingestion resolves its content source from the session's invoke config, so
# its callable is a (state, config) node — LangGraph passes both when the
# signature declares them.
type ConfiguredStateNode = Callable[[LegalAgentState, RunnableConfig], Awaitable[dict[str, Any]]]


def _ingestion_document_failure(
    failure: DocumentTextFailure,
    *,
    canonical: bool = False,
) -> dict[str, Any]:
    match failure.code:
        case DocumentTextFailureCode.NOT_FOUND:
            return _fail(
                "ingestion",
                "DOCUMENT_NOT_FOUND",
                failure.message,
            )
        case DocumentTextFailureCode.EMPTY:
            return _fail(
                "ingestion",
                "INGESTION_INCOMPLETE" if canonical else "DOCUMENT_EMPTY_TEXT",
                failure.message,
            )
        case DocumentTextFailureCode.UNAVAILABLE:
            return _fail(
                "ingestion",
                "DOCUMENT_SOURCE_UNAVAILABLE",
                failure.message,
                retryable=True,
            )
        case _ as unreachable:
            assert_never(unreachable)


def _ingestion_snapshot(
    doc_ids: tuple[str, ...],
    status: IngestionDocumentStatus,
    *,
    error_code: str | None = None,
    retryable: bool = False,
) -> list[IngestionDocumentState]:
    """Build a complete, bounded readiness snapshot for an authorized scope."""

    return [
        IngestionDocumentState(
            document_id=doc_id,
            status=status,
            error_code=error_code,
            retryable=retryable,
        )
        for doc_id in doc_ids
    ]


def _canonical_segment(chunk: CanonicalDocumentChunk) -> ClauseSegment | None:
    """Adapt one persisted chunk without changing its evidence identity."""
    content = chunk.content.strip()
    if not content:
        return None
    try:
        clause_type = ClauseType(chunk.clause_type) if chunk.clause_type else ClauseType.OTHER
    except ValueError:
        clause_type = ClauseType.OTHER
    section_ref = chunk.locus or f"chunk-{chunk.chunk_index}"
    return ClauseSegment(
        clause_id=(
            f"canonical:{chunk.document_id}:v{chunk.document_version}:chunk:{chunk.chunk_index}"
        ),
        clause_type=clause_type,
        text=chunk.content,
        section_ref=section_ref,
        start_char=0,
        end_char=len(chunk.content),
        document_id=chunk.document_id,
        document_version=chunk.document_version,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        contract_index=chunk.contract_index,
        page_no=chunk.page_no,
        chunk_kind=chunk.chunk_kind,
        metadata=dict(chunk.metadata),
        content_digest=hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
        retrieval_pointer=(
            f"canonical:{chunk.document_id}:v{chunk.document_version}:chunk:{chunk.chunk_id}"
        ),
    )


def _validate_canonical_segments(
    chunks: tuple[CanonicalDocumentChunk, ...],
    allowed_doc_ids: tuple[str, ...],
) -> tuple[ClauseSegment, ...] | None:
    """Reject malformed canonical structure instead of silently resegmenting."""
    allowed = set(allowed_doc_ids)
    seen: set[tuple[str, int, int]] = set()
    segments: list[ClauseSegment] = []
    for chunk in chunks:
        if chunk.document_id not in allowed or not chunk.chunk_id:
            return None
        identity = (chunk.document_id, chunk.document_version, chunk.chunk_index)
        if identity in seen:
            return None
        seen.add(identity)
        segment = _canonical_segment(chunk)
        if segment is None:
            return None
        segments.append(segment)
    if not segments or {segment.document_id for segment in segments} != allowed:
        return None
    return tuple(
        sorted(
            segments,
            key=lambda segment: (
                segment.document_id,
                segment.document_version,
                segment.chunk_index,
                segment.chunk_id,
            ),
        )
    )


async def _fetch_canonical_chunks(
    source: Any,
    *,
    user_id: str,
    doc_ids: tuple[str, ...],
    tenant_id: str | None,
) -> Any | None:
    """Call the canonical source seam while tolerating legacy test doubles."""
    method: Any = None
    for name in ("fetch_segments", "fetch_canonical_chunks", "fetch_chunks"):
        candidate = getattr(source, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None:
        return None
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, Any] = {"user_id": user_id}
    if "doc_ids" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        kwargs["doc_ids"] = doc_ids
    elif "allowed_doc_ids" in parameters:
        kwargs["allowed_doc_ids"] = doc_ids
    else:
        return None
    if tenant_id and (
        "tenant_id" in parameters
        or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    ):
        kwargs["tenant_id"] = tenant_id
    return await method(**kwargs)


def make_state_hydration_node() -> StateNode:
    """First node on every run: migrate persisted state before logic reads it.

    The service pre-hydrates on the UX path, so this is the defense layer for
    any future direct graph entrypoint: matching versions pass through
    unchanged, legacy shapes upgrade deterministically, and unknown versions
    raise StateSchemaVersionError — fail-closed, never a KeyError downstream.
    """

    async def state_hydration_node(state: LegalAgentState) -> dict[str, Any]:
        # cast: TypedDict is a dict at runtime; hydrate_state only reads.
        current = cast("dict[str, Any]", state)
        hydrated = hydrate_state(current)
        return {
            key: value
            for key, value in hydrated.items()
            if key not in current or current[key] != value
        }

    return state_hydration_node


def make_gateway_node() -> StateNode:
    async def gateway_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            user_id=state["user_id"],
            thread_id=state["thread_id"],
            correlation_id=state["correlation_id"],
        )

        if not state.get("doc_id"):
            log.error("gateway_missing_doc_id")
            return _fail(
                "gateway",
                "MISSING_DOC_ID",
                "doc_id is required to start the pipeline",
            )

        working_memory: dict[str, Any] = dict(state.get("working_memory", {}))
        working_memory["gateway_validated"] = True
        working_memory["session_start_ts"] = _utc_now_iso()

        log.info("gateway_validated")

        return {
            "working_memory": working_memory,
            "status": WorkflowStatus.QNA_CLARIFICATION,
        }

    return gateway_node


def make_qna_node(qna_llm: Runnable[list[Any], QnAOutput]) -> StateNode:
    async def qna_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="qna",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
        )

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        messages = [SystemMessage(content=_QNA_SYSTEM_PROMPT), *state["messages"]]
        result: QnAOutput = await qna_llm.ainvoke(messages)
        ledger, blocked = _charge_model_result(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

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
                "budget_ledger": ledger,
                "messages": [HumanMessage(content=answer_text)],
                "qna_confidence": result.confidence,
                "status": WorkflowStatus.QNA_CLARIFICATION,
            }

        working_memory = dict(state.get("working_memory", {}))
        working_memory["clarified_intent"] = result.intent
        if result.document_type_guess:
            working_memory["document_type"] = result.document_type_guess

        return {
            "budget_ledger": ledger,
            "qna_confidence": result.confidence,
            "working_memory": working_memory,
            "status": WorkflowStatus.PLAN_PENDING,
        }

    return qna_node


def route_after_qna(state: LegalAgentState) -> str:
    if state["status"] == WorkflowStatus.QNA_CLARIFICATION:
        return "qna"
    return "orchestrator"


def _orchestrator_capability_context(
    state: LegalAgentState,
    ledger: BudgetLedger,
) -> dict[str, object]:
    """Build deterministic, server-owned context for the structured router.

    The orchestrator intentionally remains a structured-output chain rather
    than gaining arbitrary model-callable tools.  This typed context is the
    narrow equivalent: readiness is derived from the authorized document
    allowlist, while delegation targets and remaining budget are bounded by
    the graph/runtime.  No identity, tenant, or document authority is read
    from model output.
    """

    allowed_doc_ids = tuple(dict.fromkeys(state.get("allowed_doc_ids", ())))
    by_document: dict[str, IngestionDocumentState] = {}
    for raw in state.get("ingestion_state", []):
        try:
            readiness = (
                raw
                if isinstance(raw, IngestionDocumentState)
                else IngestionDocumentState.model_validate(raw)
            )
        except ValidationError:
            continue
        if readiness.document_id in allowed_doc_ids:
            by_document[readiness.document_id] = readiness

    documents = [
        by_document.get(
            document_id,
            IngestionDocumentState(
                document_id=document_id,
                status=IngestionDocumentStatus.NOT_READY,
            ),
        ).model_dump(mode="json")
        for document_id in allowed_doc_ids
    ]
    return {
        "documents": documents,
        "delegation": {
            "allowed_targets": sorted(_VALID_WORKER_NODES),
            "remaining_model_calls": max(0, ledger.limits.model_calls - ledger.model_calls),
            "remaining_tool_calls": max(0, ledger.limits.tool_calls - ledger.tool_calls),
        },
    }


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

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        if state["status"] == WorkflowStatus.FINALIZING:
            return {
                "orchestrator_action": OrchestratorAction(
                    action_type=OrchestratorActionType.SYNTHESIZE,
                    reflection="Human review completed; finalize the reviewed analysis.",
                ),
                "current_step": state["current_step"] + 1,
                "budget_ledger": ledger,
            }
        if state["status"] in {WorkflowStatus.FAILED, WorkflowStatus.COMPLETED}:
            return {
                "orchestrator_action": OrchestratorAction(
                    action_type=OrchestratorActionType.DONE,
                    reflection="The workflow is terminal; do not schedule more workers.",
                ),
                "current_step": state["current_step"] + 1,
                "budget_ledger": ledger,
            }

        risk_analysis = state["risk_analysis"]
        compliance_result = state["compliance_result"]
        grounding = state["grounding"]
        human_review = state["human_review"]
        context = {
            "workflow_status": state["status"].value,
            "current_step": state["current_step"],
            "plan_approved": state.get("plan_approved", False),
            "plan": [step.model_dump(mode="json") for step in state.get("plan", [])],
            "errors": [error.model_dump(mode="json") for error in state.get("errors", [])],
            "has_document_text": bool(state.get("document_text")),
            "has_normalized_document": state.get("normalized_document") is not None,
            "segment_count": len(state.get("segments", [])),
            "risk_summary": risk_analysis.summary if risk_analysis is not None else None,
            "compliance_summary": (
                compliance_result.summary if compliance_result is not None else None
            ),
            "grounding_verified": grounding.verified if grounding is not None else None,
            "human_review_approved": human_review.approved if human_review is not None else None,
            # Deterministic capability context: the model receives readiness
            # and bounded delegation limits, but no tool can widen scope.
            "capabilities": _orchestrator_capability_context(state, ledger),
        }

        messages = [
            SystemMessage(content=_ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(context, ensure_ascii=False)),
            *state["messages"],
        ]
        action: OrchestratorAction = await orchestrator_llm.ainvoke(messages)
        ledger, blocked = _charge_model_result(ledger, action)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        if (
            action.action_type == OrchestratorActionType.CONTINUE
            and action.target_node not in _VALID_WORKER_NODES
        ):
            log.error("orchestrator_invalid_target", target=action.target_node)
            return _fail(
                node="orchestrator",
                code="INVALID_TARGET_NODE",
                message=f"Orchestrator routed to unknown node: {action.target_node}",
            )

        log.info(
            "orchestrator_action_decided",
            action_type=action.action_type,
            target=action.target_node,
            reflection=action.reflection[:_REFLECTION_LOG_CHARS],
        )

        return {
            "orchestrator_action": action,
            "budget_ledger": ledger,
            "current_step": state["current_step"] + 1,
        }

    return orchestrator_node


def route_from_orchestrator(state: LegalAgentState) -> str:
    status = state.get("status")
    if status in {WorkflowStatus.FAILED, WorkflowStatus.COMPLETED}:
        return END
    if status == WorkflowStatus.FINALIZING:
        return "finalization"

    action = state.get("orchestrator_action")
    if action is None:
        return "planner_generate"

    if status == WorkflowStatus.PLAN_REJECTED:
        return "planner_generate"

    match action.action_type:
        case OrchestratorActionType.START_PIPELINE:
            if not state.get("plan") or not state.get("plan_approved", False):
                return "planner_generate"
            return "ingestion"
        case OrchestratorActionType.CONTINUE:
            return action.target_node or "ingestion"
        case OrchestratorActionType.SYNTHESIZE:
            return "finalization"
        case OrchestratorActionType.DONE:
            return END
        case _:
            return "planner_generate"


def make_plan_generation_node(planner_llm: Runnable[list[Any], PlannerOutput]) -> StateNode:
    """Generate and checkpoint the proposal before entering HITL.

    ``interrupt`` resumes from the beginning of the node that contains it.
    Keeping model generation in a separate node means an approval replay only
    re-enters the deterministic approval node and can never silently replace
    the proposal the reviewer saw.
    """

    async def plan_generation_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="planner_generate",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
        )

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
        ]
        rejection_feedback = state.get("plan_rejection_feedback")
        if rejection_feedback:
            # The feedback is an ephemeral planning input: it is deliberately
            # sent to this LLM call without adding it to transcript messages
            # or logging its content.
            messages.append(
                HumanMessage(
                    content=(
                        "The reviewer rejected the previous plan. Address this feedback "
                        f"when generating the replacement plan:\n{rejection_feedback}"
                    )
                )
            )
        messages.extend(state["messages"])
        result: PlannerOutput = await planner_llm.ainvoke(messages)
        ledger, blocked = _charge_model_result(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        log.info("planner_plan_generated", step_count=len(result.steps))

        working_memory = dict(state.get("working_memory", {}))
        working_memory["plan_rationale"] = result.rationale
        return {
            "plan": list(result.steps),
            "current_step": 0,
            "plan_approved": False,
            "plan_rejection_feedback": None,
            "working_memory": working_memory,
            "status": WorkflowStatus.PLAN_AWAITING_APPROVAL,
            "budget_ledger": ledger,
        }

    return plan_generation_node


def make_plan_approval_node() -> StateNode:
    """Show the checkpointed proposal and apply the human decision.

    This node intentionally contains no model call.  On ``Command(resume=…)``
    LangGraph replays this node from its beginning, but the proposal has
    already been persisted by ``make_plan_generation_node`` and is therefore
    exactly the one that was shown to the reviewer.
    """

    async def plan_approval_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="planner",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
        )

        proposal = list(state.get("plan", []))
        if not proposal:
            return _fail(
                "planner",
                "PLAN_MISSING",
                "Cannot request approval for an empty execution plan",
            )

        human_response: dict[str, Any] = interrupt(
            {
                "type": HITLInterruptType.PLAN_APPROVAL,
                "plan": [step.model_dump(mode="json") for step in proposal],
                "rationale": state.get("working_memory", {}).get("plan_rationale", ""),
                "message": "Please review and approve the execution plan",
            }
        )

        action: str = human_response.get("action", "approve")

        if action == "reject":
            log.info("planner_plan_rejected")
            feedback = human_response.get("feedback")
            rejection_feedback = feedback if isinstance(feedback, str) else None
            return {
                **_fail(
                    "planner",
                    "PLAN_REJECTED",
                    rejection_feedback or "Plan rejected by reviewer",
                    status=WorkflowStatus.PLAN_REJECTED,
                    retryable=True,
                ),
                "plan_rejection_feedback": rejection_feedback,
            }

        if action == "modify":
            raw_steps = human_response.get("modified_plan") or []
            if not isinstance(raw_steps, list):
                return _fail(
                    "planner",
                    "INVALID_MODIFIED_PLAN",
                    "The modified execution plan must be a list of plan steps",
                )
            approved_plan = [PlanStep.model_validate(step) for step in raw_steps] or proposal
            log.info("planner_plan_modified", step_count=len(approved_plan))
        else:
            approved_plan = proposal

        return {
            "plan": list(approved_plan),
            "current_step": 0,
            "plan_approved": True,
            "plan_rejection_feedback": None,
            "status": WorkflowStatus.PLAN_APPROVED,
        }

    return plan_approval_node


def make_planner_node(planner_llm: Runnable[list[Any], PlannerOutput]) -> StateNode:
    """Backward-compatible planner factory.

    New graph topology uses :func:`make_plan_generation_node` and
    :func:`make_plan_approval_node` explicitly.  Keeping this name available
    avoids breaking callers that imported the pre-split factory; it now means
    the generation half only, with approval deliberately composed by the graph
    builder.
    """

    return make_plan_generation_node(planner_llm)


def make_ingestion_node() -> ConfiguredStateNode:
    """The ingestion node feeds normalization from the canonical text source.

    The injected ``DocumentContentSource`` rides the session's
    ``RunnableConfig.configurable`` (see ``document_source.py``). Expected
    document-state failures return clean state errors — a node must never
    raise for them, and must never silently continue with empty text.
    """

    async def ingestion_node(state: LegalAgentState, config: RunnableConfig) -> dict[str, Any]:  # noqa: PLR0912, PLR0914
        allowed_doc_ids = tuple(dict.fromkeys(state.get("allowed_doc_ids") or [state["doc_id"]]))
        log = logger.bind(
            node="ingestion",
            doc_id=state["doc_id"],
            user_id=state["user_id"],
            allowed_doc_ids=allowed_doc_ids,
        )
        log.info("ingestion_started")

        retry_count = state.get("retry_count", 0)
        if retry_count >= _MAX_RETRIES:
            return _fail(
                "ingestion",
                "MAX_RETRIES_EXCEEDED",
                f"Ingestion failed after {_MAX_RETRIES} attempts",
            ) | {
                "ingestion_state": _ingestion_snapshot(
                    allowed_doc_ids,
                    IngestionDocumentStatus.FAILED,
                    error_code="MAX_RETRIES_EXCEEDED",
                )
            }

        source = document_content_source_from_config(config)
        if source is None:
            log.error("ingestion_document_source_not_wired")
            return _fail(
                "ingestion",
                "DOCUMENT_SOURCE_NOT_WIRED",
                "No document content source was injected for this run",
            ) | {
                "ingestion_state": _ingestion_snapshot(
                    allowed_doc_ids,
                    IngestionDocumentStatus.NOT_READY,
                    error_code="DOCUMENT_SOURCE_NOT_WIRED",
                    retryable=True,
                )
            }

        tenant_id = state.get("tenant_id") or None
        try:
            canonical_fetch = await _fetch_canonical_chunks(
                source,
                user_id=state["user_id"],
                doc_ids=allowed_doc_ids,
                tenant_id=tenant_id,
            )
        except Exception as exc:
            exc.add_note(f"doc_ids={','.join(allowed_doc_ids)}, operation=ingestion_fetch_segments")
            log.exception("canonical_document_source_failed")
            return _fail(
                "ingestion",
                "DOCUMENT_SOURCE_UNAVAILABLE",
                "Canonical document content provider failed unexpectedly",
                retryable=True,
            )

        if canonical_fetch is not None:
            if isinstance(canonical_fetch, Failure):
                failure = canonical_fetch.failure()
                return _ingestion_document_failure(failure, canonical=True) | {
                    "ingestion_state": _ingestion_snapshot(
                        allowed_doc_ids,
                        IngestionDocumentStatus.FAILED,
                        error_code=str(failure.code),
                        retryable=failure.code is DocumentTextFailureCode.UNAVAILABLE,
                    )
                }
            chunks = canonical_fetch.unwrap()
            segments = _validate_canonical_segments(chunks, allowed_doc_ids)
            if segments is None:
                return _fail(
                    "ingestion",
                    "INGESTION_INCOMPLETE",
                    "Selected documents do not contain usable canonical chunk structure",
                ) | {
                    "ingestion_state": _ingestion_snapshot(
                        allowed_doc_ids,
                        IngestionDocumentStatus.FAILED,
                        error_code="INGESTION_INCOMPLETE",
                    )
                }
            scope_versions = tuple(
                (document_id, version, None)
                for document_id, version in sorted(
                    {(segment.document_id, segment.document_version) for segment in segments}
                )
            )
            contract_values: dict[tuple[str, int], list[int | None]] = {}
            for segment in segments:
                contract_values.setdefault(
                    (segment.document_id, segment.document_version), []
                ).append(segment.contract_index)
            scope_contracts = tuple(
                (
                    document_id,
                    version,
                    tuple(sorted({index for index in indexes if index is not None})),
                    bool(indexes) and all(index is None for index in indexes),
                )
                for (document_id, version), indexes in sorted(contract_values.items())
            )
            # ``segments`` is an append-only LangGraph channel because entity
            # extraction fans out with Send. Avoid duplicating canonical rows
            # when a paused legacy checkpoint replays the ingestion node.
            existing_ids = {segment.clause_id for segment in state.get("segments", [])}
            new_segments = [
                segment for segment in segments if segment.clause_id not in existing_ids
            ]
            log.info("canonical_ingestion_completed", segment_count=len(segments))
            return {
                "canonical_segments": list(segments),
                "canonical_structure_ready": True,
                "legacy_structure_fallback": False,
                "document_text": None,
                "segments": new_segments,
                "document_versions": list(scope_versions),
                "document_contracts": list(scope_contracts),
                "ingestion_state": _ingestion_snapshot(
                    allowed_doc_ids,
                    IngestionDocumentStatus.READY,
                ),
                "status": WorkflowStatus.NORMALIZING,
            }

        if len(allowed_doc_ids) > 1:
            # A legacy source can only return one anonymous text blob. Refuse
            # it for a multi-document scope rather than losing boundaries.
            return _fail(
                "ingestion",
                "INGESTION_INCOMPLETE",
                "Multi-document analysis requires a canonical chunk source",
            ) | {
                "ingestion_state": _ingestion_snapshot(
                    allowed_doc_ids,
                    IngestionDocumentStatus.NOT_READY,
                    error_code="INGESTION_INCOMPLETE",
                )
            }

        try:
            fetch_kwargs: dict[str, str] = {
                "user_id": state["user_id"],
                "doc_id": state["doc_id"],
            }
            if state.get("tenant_id"):
                fetch_kwargs["tenant_id"] = state["tenant_id"]
            fetch = await source.fetch_text(**fetch_kwargs)
        except Exception as exc:  # noqa: BLE001 — node boundary: a provider must not abort the WS turn with a raise
            exc.add_note(f"doc_id={state['doc_id']}, operation=ingestion_fetch_document_text")
            logger.bind(
                node="ingestion", doc_id=state["doc_id"], user_id=state["user_id"]
            ).exception("Document content provider leaked a failure")
            return _fail(
                "ingestion",
                "DOCUMENT_SOURCE_UNAVAILABLE",
                "Document content provider failed unexpectedly",
                retryable=True,
            ) | {
                "ingestion_state": _ingestion_snapshot(
                    allowed_doc_ids,
                    IngestionDocumentStatus.FAILED,
                    error_code="DOCUMENT_SOURCE_UNAVAILABLE",
                    retryable=True,
                )
            }

        if isinstance(fetch, Failure):
            error = fetch.failure()
            log.warning(
                "ingestion_document_unreadable",
                failure_code=str(error.code),
                failure_message=error.message,
            )
            return _ingestion_document_failure(error) | {
                "ingestion_state": _ingestion_snapshot(
                    allowed_doc_ids,
                    IngestionDocumentStatus.FAILED,
                    error_code=str(error.code),
                    retryable=error.code is DocumentTextFailureCode.UNAVAILABLE,
                )
            }

        text: str = fetch.unwrap()
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
            return _fail(
                "ingestion",
                "LOW_OCR_CONFIDENCE",
                f"OCR confidence {confidence:.0%} - user declined re-upload",
            )

        log.info("ingestion_completed", text_length=len(text), confidence=confidence)
        return {
            "document_text": text,
            "canonical_segments": [],
            "canonical_structure_ready": False,
            "legacy_structure_fallback": True,
            "ingestion_state": _ingestion_snapshot(
                allowed_doc_ids,
                IngestionDocumentStatus.WARNINGS,
            ),
            "status": WorkflowStatus.NORMALIZING,
        }

    return ingestion_node


def make_normalization_node(
    normalization_llm: Runnable[list[Any], NormalizedDocument],
) -> StateNode:
    async def normalization_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="normalization", doc_id=state["doc_id"])

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        if state.get("canonical_structure_ready"):
            segments = state.get("canonical_segments", [])
            if not segments:
                return _fail(
                    "normalization",
                    "INGESTION_INCOMPLETE",
                    "Canonical ingestion reported ready without any canonical segments",
                )
            log.info("normalization_bypassed_canonical_structure", segment_count=len(segments))
            return {"status": WorkflowStatus.SEGMENTING}

        if not state.get("document_text"):
            return _fail(
                "normalization",
                "MISSING_DOCUMENT_TEXT",
                "document_text not populated by ingestion",
            )

        try:
            result: NormalizedDocument = await _invoke_structured(
                normalization_llm,
                _NORMALIZATION_SYSTEM_PROMPT,
                state["document_text"],
            )
        except OutputParserException as exc:
            exc.add_note(f"node=normalization, doc_id={state['doc_id']}")
            log.bind(error=str(exc)).exception("normalization_output_invalid")
            return _fail(
                "normalization",
                "INVALID_NORMALIZED_DOCUMENT",
                "The model returned document structure that did not match the required schema",
                retryable=True,
            )
        ledger, blocked = _charge_model_result(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        log.info("normalization_completed", section_count=len(result.sections))

        return {
            "normalized_document": result,
            "budget_ledger": ledger,
            "status": WorkflowStatus.SEGMENTING,
        }

    return normalization_node


def make_segmentation_node(
    segmentation_llm: Runnable[list[Any], ClauseSegmentationOutput],
) -> StateNode:
    async def segmentation_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="segmentation", doc_id=state["doc_id"])

        if state.get("canonical_structure_ready"):
            segments = state.get("canonical_segments", [])
            if not segments:
                return _fail(
                    "segmentation",
                    "INGESTION_INCOMPLETE",
                    "Canonical ingestion reported ready without any canonical segments",
                )
            log.info("segmentation_bypassed_canonical_structure", segment_count=len(segments))
            # Canonical segmentation is a deterministic projection of the
            # already-ingested chunks; no model invocation occurred here, so
            # it must not consume a model-call budget unit per chunk.
            return {"status": WorkflowStatus.EXTRACTING_ENTITIES}

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        normalized_document = state["normalized_document"]
        if normalized_document is None:
            return _fail(
                "segmentation",
                "MISSING_NORMALIZED_DOCUMENT",
                "normalized_document not populated by normalization node",
            )

        doc_text = "\n".join(s.content for s in normalized_document.sections)
        try:
            result: ClauseSegmentationOutput = await _invoke_structured(
                segmentation_llm,
                _SEGMENTATION_SYSTEM_PROMPT,
                doc_text,
            )
        except OutputParserException as exc:
            exc.add_note(f"node=segmentation, doc_id={state['doc_id']}")
            log.bind(error=str(exc)).exception("segmentation_output_invalid")
            return _fail(
                "segmentation",
                "INVALID_CLAUSE_SEGMENTATION",
                "The model returned clause segments that did not match the required schema",
                retryable=True,
            )
        ledger, blocked = _charge_model_result(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        log.info("segmentation_completed", segment_count=len(result.segments))

        try:
            ledger = ledger.consume(BudgetDimension.MODEL_CALLS, len(result.segments))
        except BudgetExhaustedError as exc:
            return {
                **_fail("budget", "BUDGET_EXHAUSTED", str(exc), retryable=False),
                "budget_ledger": ledger.model_copy(
                    update={"exhausted": True, "exhausted_dimension": exc.dimension}
                ),
            }

        return {
            "segments": list(result.segments),
            "budget_ledger": ledger,
            "status": WorkflowStatus.EXTRACTING_ENTITIES,
        }

    return segmentation_node


def dispatch_entity_extraction(state: LegalAgentState) -> list[Send]:
    if state.get("status") == WorkflowStatus.FAILED:
        return []
    document_context: dict[str, Any] = {
        "jurisdiction": state["working_memory"].get("jurisdiction", "India"),
        "document_type": state["working_memory"].get("document_type", "unknown"),
        "doc_id": state["doc_id"],
        "user_id": state["user_id"],
        "tenant_id": state.get("tenant_id", ""),
        "allowed_doc_ids": tuple(state.get("allowed_doc_ids", (state["doc_id"],))),
    }
    return [
        Send(
            "entity_extraction",
            # Keep the live Pydantic object on the in-memory Send payload so
            # the active super-step can use its just-read text.  Calling
            # ``model_dump`` here would apply ClauseSegment's checkpoint
            # exclusion and make every same-run canonical extraction appear
            # unavailable.  The object is never written back to graph state;
            # only its identity fields survive checkpoint serialization.
            {
                "clause": segment,
                "document_context": document_context,
            },
        )
        for segment in state["segments"]
    ]


def make_entity_extraction_node(
    entity_llm: Runnable[list[Any], EntityExtractionOutput],
) -> Callable[[dict[str, Any], RunnableConfig | None], Awaitable[dict[str, Any]]]:
    async def entity_extraction_node(
        state: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        clause = ClauseSegment.model_validate(state["clause"])
        doc_context: dict[str, Any] = state.get("document_context", {})

        log = logger.bind(node="entity_extraction", clause_id=clause.clause_id)

        # A resumed checkpoint intentionally hydrates canonical segments with
        # identity metadata only.  Re-read the exact chunk through the
        # server-injected, user/tenant-scoped source when available.  Never
        # let an empty dereference be interpreted as an empty clause.
        if not clause.text:
            source = document_content_source_from_config(config) if config is not None else None
            if source is not None and clause.chunk_id:
                try:
                    fetched = await _fetch_canonical_chunks(
                        source,
                        user_id=str(doc_context.get("user_id", "")),
                        doc_ids=tuple(
                            str(doc_id)
                            for doc_id in doc_context.get("allowed_doc_ids", (clause.document_id,))
                        ),
                        tenant_id=str(doc_context.get("tenant_id", "")) or None,
                    )
                except Exception as exc:  # noqa: BLE001 — node boundary fails closed on provider errors
                    exc.add_note(
                        f"node=entity_extraction, chunk_id={clause.chunk_id}, operation=dereference"
                    )
                    fetched = None
                if isinstance(fetched, Success):
                    matching = next(
                        (chunk for chunk in fetched.unwrap() if chunk.chunk_id == clause.chunk_id),
                        None,
                    )
                    if matching is not None:
                        clause = clause.model_copy(update={"text": matching.content})
            if not clause.text:
                return _fail(
                    "entity_extraction",
                    "EVIDENCE_UNAVAILABLE",
                    "Canonical clause text is unavailable for this resumed run",
                    retryable=True,
                )

        prompt = (
            f"Document type: {doc_context.get('document_type', 'unknown')}\n"
            f"Jurisdiction: {doc_context.get('jurisdiction', 'India')}\n"
            f"Document ID: {clause.document_id}\n"
            f"Document version: {clause.document_version}\n"
            f"Canonical chunk ID: {clause.chunk_id}\n"
            f"Page: {clause.page_no}\n"
            f"Clause ID: {clause.clause_id}\n"
            f"Clause type: {clause.clause_type}\n\n"
            f"Contract index: {clause.contract_index}\n\n"
            f"Clause text:\n{clause.text}"
        )
        messages = [
            SystemMessage(content=_ENTITY_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        result: EntityExtractionOutput = await entity_llm.ainvoke(messages)
        log.info("entity_extraction_done", entity_count=len(result.entities))
        entities = [
            entity.model_copy(
                update={
                    "contract_index": (
                        entity.contract_index
                        if entity.contract_index is not None
                        else clause.contract_index
                    ),
                    "citation": entity.citation.model_copy(
                        update={
                            "contract_index": (
                                entity.citation.contract_index
                                if entity.citation.contract_index is not None
                                else clause.contract_index
                            )
                        }
                    ),
                }
            )
            for entity in result.entities
        ]

        # Segmentation reserves one persisted model-call unit per dispatched
        # clause before this parallel fan-out.  Workers intentionally do not
        # write the shared ledger: concurrent LastValue writes would race and
        # either fail the graph update or lose usage on resume.
        return {"extracted_entities": entities}

    return entity_extraction_node


def make_relationship_mapping_node(
    relationship_llm: Runnable[list[Any], RelationshipMappingOutput],
) -> StateNode:
    async def relationship_mapping_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="relationship_mapping",
            entity_count=len(state["extracted_entities"]),
        )

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        entity_summary = "\n".join(
            f"[{e.entity_type}] {e.value} (clause: {e.clause_id}, "
            f"contract: {e.contract_index if e.contract_index is not None else e.citation.contract_index}, "
            f"party: {e.party or 'N/A'})"
            for e in state["extracted_entities"]
        )
        result: RelationshipMappingOutput = await _invoke_structured(
            relationship_llm,
            _RELATIONSHIP_MAPPING_SYSTEM_PROMPT,
            entity_summary,
        )
        ledger, blocked = _charge_model_result(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        log.info("relationship_mapping_done", relationship_count=len(result.relationships))

        relationships = [
            relationship.model_copy(
                update={
                    "contract_index": (
                        relationship.contract_index
                        if relationship.contract_index is not None
                        else relationship.citation.contract_index
                    )
                }
            )
            for relationship in result.relationships
        ]
        return {
            "relationships": relationships,
            "budget_ledger": ledger,
            "status": WorkflowStatus.ANALYZING_RISKS,
        }

    return relationship_mapping_node


def delegation_pairs_from(messages: list[Any]) -> list[tuple[Any, Any]]:
    """Extract delegation pairs from a sub-agent result's message list.

    Scans for assistant messages invoking ``transfer_to_*`` tools and rebuilds
    each as a linked pair via the shared constructor. The sub-agent's internal
    loop stays private; only the delegation becomes visible in shared state.
    """
    pairs: list[tuple[Any, Any]] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name", "") if isinstance(call, dict) else ""
            if not name.startswith("transfer_to_"):
                continue
            args = call.get("args", {}) if isinstance(call, dict) else {}
            reason = args.get("reason", "") if isinstance(args, dict) else ""
            pairs.append(make_delegation_pair(name.removeprefix("transfer_to_"), str(reason)))
    return pairs


def _analysis_tool_context(state: LegalAgentState) -> ToolRuntimeContext:
    """Project only server-owned scope into a nested specialist graph."""
    run_id = state.get("run_id") or state.get("correlation_id") or state["thread_id"]
    evidence_checkpoint_id = state.get("evidence_checkpoint_id") or f"run:{run_id}"
    return {
        "user_id": state["user_id"],
        "tenant_id": state.get("tenant_id", ""),
        "allowed_doc_ids": tuple(state.get("allowed_doc_ids", ())),
        "allowed_operations": tuple(state.get("allowed_operations", ())),
        "capabilities": tuple(state.get("capabilities", ())),
        "thread_id": state["thread_id"],
        "step_id": str(state.get("current_step", 0)),
        "run_id": run_id,
        "checkpoint_id": evidence_checkpoint_id,
        "document_versions": tuple(state.get("document_versions", ())),
        "document_contracts": tuple(state.get("document_contracts", ())),
    }


async def _invoke_analysis_agent(
    agent: Any,
    payload: dict[str, Any],
    *,
    context: ToolRuntimeContext,
) -> Any:
    """Invoke a nested agent with scope context, preserving narrow test doubles.

    ``CompiledStateGraph.ainvoke`` accepts the ``context=`` keyword.  Small
    unit-test doubles and legacy adapters often expose only ``ainvoke(input)``;
    they do not execute tools and should not be forced to implement the
    LangGraph runtime surface merely to test node result mapping.
    """
    try:
        parameters = inspect.signature(agent.ainvoke).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_context = "context" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if accepts_context:
        return await agent.ainvoke(payload, context=context)
    return await agent.ainvoke(payload)


def make_risk_analysis_node(risk_agent: Any) -> StateNode:
    async def risk_analysis_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="risk_analysis",
            clause_count=len(state["segments"]),
            entity_count=len(state["extracted_entities"]),
        )

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        context = _build_analysis_context(state)
        evidence_registry: dict[str, EvidenceRef] = dict(state.get("evidence_registry", {}))
        coverage = dict(state.get("coverage", {}))
        try:
            result = await _invoke_analysis_agent(
                risk_agent,
                {
                    "messages": [
                        SystemMessage(content=_RISK_ANALYSIS_SYSTEM_PROMPT),
                        HumanMessage(content=context),
                        *state["messages"],
                    ]
                },
                context=_analysis_tool_context(state),
            )
        except Exception as exc:
            exc.add_note(f"node=risk_analysis, doc_id={state['doc_id']}")
            log.exception("risk_agent_unavailable")
            return {
                "risk_analysis": _not_assessed_risk_output(
                    "Risk analysis could not be completed because the analysis agent was unavailable."
                ),
                "status": WorkflowStatus.CHECKING_COMPLIANCE,
                "errors": [
                    _analysis_failure(
                        "risk_analysis",
                        "RISK_AGENT_UNAVAILABLE",
                        "Risk analysis was not assessed because the analysis agent was unavailable.",
                    )
                ],
            }

        ledger, blocked = _consume_observed_tool_calls(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        ledger, blocked = _consume_observed_token_usage(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        structured = _agent_structured_response(result, RiskAnalysisOutput)
        if structured is None:
            log.error("risk_agent_missing_structured_response")
            risk_output = _not_assessed_risk_output(
                "Risk analysis was not assessed because no structured result was returned."
            )
            errors = [
                _analysis_failure(
                    "risk_analysis",
                    "RISK_STRUCTURED_OUTPUT_MISSING",
                    "Risk analysis returned no validated structured result.",
                )
            ]
        else:
            risk_output = cast("RiskAnalysisOutput", structured)
            errors = []
            evidence_registry, coverage = _merge_evidence_metadata(state, result)

        log.info(
            "risk_analysis_completed",
            finding_count=len(risk_output.findings),
            overall_label=risk_output.overall_label,
        )
        delegation_messages = [
            message for pair in delegation_pairs_from(_agent_messages(result)) for message in pair
        ]

        return {
            "risk_analysis": risk_output,
            "status": WorkflowStatus.CHECKING_COMPLIANCE,
            "messages": delegation_messages,
            "errors": errors,
            "evidence_registry": evidence_registry,
            "coverage": coverage,
            "budget_ledger": ledger,
        }

    return risk_analysis_node


def _build_analysis_context(state: LegalAgentState) -> str:
    clauses = "\n".join(
        (
            f"[{seg.clause_type}] {seg.clause_id} "
            f"(document={seg.document_id or state['doc_id']}, "
            f"version={seg.document_version}, chunk={seg.chunk_id or 'legacy'}, "
            f"page={seg.page_no}): {seg.text[:_CLAUSE_CONTEXT_CHARS]}"
        )
        for seg in state["segments"]
    )
    entities = "\n".join(
        f"{e.entity_type}: {e.value} (party: {e.party or 'N/A'})"
        for e in state["extracted_entities"]
    )
    relationships = "\n".join(
        f"{r.from_node} -> {r.relationship} -> {r.to_node} "
        f"(clause: {r.clause_id}, contract: {r.contract_index})"
        for r in state["relationships"]
    )
    return f"CLAUSES:\n{clauses}\n\nENTITIES:\n{entities}\n\nRELATIONSHIPS:\n{relationships}"


def _not_assessed_risk_output(reason: str) -> RiskAnalysisOutput:
    return RiskAnalysisOutput(
        findings=[],
        overall_label=None,
        summary=reason,
        assessment_status=AssessmentStatus.NOT_ASSESSED,
    )


def make_compliance_node(compliance_agent: Any) -> StateNode:
    async def compliance_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="compliance")

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        context = _build_analysis_context(state)
        evidence_registry: dict[str, EvidenceRef] = dict(state.get("evidence_registry", {}))
        coverage = dict(state.get("coverage", {}))
        result: Any = None
        try:
            result = await _invoke_analysis_agent(
                compliance_agent,
                {
                    "messages": [
                        SystemMessage(content=_COMPLIANCE_SYSTEM_PROMPT),
                        HumanMessage(content=context),
                        *state["messages"],
                    ]
                },
                context=_analysis_tool_context(state),
            )
        except (StructuredOutputValidationError, OutputParserException) as exc:
            exc.add_note(f"node=compliance, doc_id={state['doc_id']}")
            log.bind(error=str(exc)).exception("compliance_output_invalid")
            compliance_output = _not_assessable_compliance_output(
                "Compliance could not be assessed because the agent returned invalid structured output."
            )
            errors = [
                _analysis_failure(
                    "compliance",
                    "INVALID_COMPLIANCE_OUTPUT",
                    "The compliance agent returned no validated structured result.",
                )
            ]
            messages: list[Any] = []
        except Exception as exc:
            exc.add_note(f"node=compliance, doc_id={state['doc_id']}")
            log.exception("compliance_agent_unavailable")
            compliance_output = _not_assessable_compliance_output(
                "Compliance could not be assessed because the analysis agent was unavailable."
            )
            errors = [
                _analysis_failure(
                    "compliance",
                    "COMPLIANCE_AGENT_UNAVAILABLE",
                    "Compliance was not assessable because the analysis agent was unavailable.",
                )
            ]
            messages = []
        ledger, blocked = _consume_observed_tool_calls(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        ledger, blocked = _consume_observed_token_usage(ledger, result)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        if result is not None:
            structured = _agent_structured_response(result, ComplianceOutput)
            if structured is None:
                log.error("compliance_agent_missing_structured_response")
                compliance_output = _not_assessable_compliance_output(
                    "Compliance could not be assessed because no structured result was returned."
                )
                errors = [
                    _analysis_failure(
                        "compliance",
                        "COMPLIANCE_STRUCTURED_OUTPUT_MISSING",
                        "Compliance returned no validated structured result.",
                    )
                ]
            else:
                compliance_output = cast("ComplianceOutput", structured)
                errors = []
                evidence_registry, coverage = _merge_evidence_metadata(state, result)
            messages = [
                message
                for pair in delegation_pairs_from(_agent_messages(result))
                for message in pair
            ]

        log.info(
            "compliance_completed",
            finding_count=len(compliance_output.findings),
            compliance_status=compliance_output.compliance_status,
        )

        return {
            "compliance_result": compliance_output,
            "status": (WorkflowStatus.FAILED if errors else WorkflowStatus.VERIFYING_GROUNDING),
            "messages": messages,
            "errors": errors,
            "evidence_registry": evidence_registry,
            "coverage": coverage,
            "budget_ledger": ledger,
        }

    return compliance_node


def _not_assessable_compliance_output(reason: str) -> ComplianceOutput:
    return ComplianceOutput(
        findings=[],
        jurisdiction="India",
        overall_compliant=None,
        summary=reason,
        compliance_status=ComplianceStatus.NOT_ASSESSABLE,
    )


async def _verify_quotes_against_canonical_source(  # noqa: PLR0914 — bounded quote re-read and span validation
    state: LegalAgentState,
    config: RunnableConfig | None,
) -> list[str]:
    """Re-read authorized canonical chunks before accepting quoted claims."""

    risk = state.get("risk_analysis")
    compliance = state.get("compliance_result")
    findings = (risk.findings if risk is not None else []) + (
        compliance.findings if compliance is not None else []
    )
    citations = [
        citation
        for finding in findings
        for citation in finding.citations
        if citation.quoted_text or citation.quote_digest
    ]
    if not citations:
        return []
    source = document_content_source_from_config(config) if config is not None else None
    if source is None:
        return ["Quoted evidence could not be re-read under the current authorization context."]
    registry = state.get("evidence_registry", {})
    refs = [registry.get(citation.evidence_id or citation.source) for citation in citations]
    doc_ids = tuple(dict.fromkeys(ref.document_id for ref in refs if isinstance(ref, EvidenceRef)))
    if not doc_ids:
        return ["Quoted evidence has no authorized canonical document reference."]
    try:
        fetched = await _fetch_canonical_chunks(
            source,
            user_id=state["user_id"],
            doc_ids=doc_ids,
            tenant_id=state.get("tenant_id") or None,
        )
    except Exception as exc:  # noqa: BLE001 — grounding fails closed on provider errors
        exc.add_note("operation=grounding_quote_reread")
        return ["Canonical evidence could not be re-read for quote verification."]
    if not isinstance(fetched, Success):
        return ["Canonical evidence could not be re-read for quote verification."]
    chunks = {chunk.chunk_id: chunk for chunk in fetched.unwrap()}
    issues: list[str] = []
    for citation, ref in zip(citations, refs, strict=True):
        if not isinstance(ref, EvidenceRef):
            issues.append(f"Quote citation {citation.source} has no valid evidence handle.")
            continue
        chunk = chunks.get(ref.chunk_id)
        if chunk is None:
            issues.append(f"Quote citation {ref.evidence_id} has no current canonical chunk.")
            continue
        quote_value = citation.quoted_text
        if quote_value is not None:
            quote = quote_value.strip()
            normalized_quote = " ".join(quote.split())
            normalized_content = " ".join(chunk.content.split())
            valid_quote = quote in chunk.content or normalized_quote in normalized_content
        else:
            valid_quote = citation.quote_digest in _canonical_quote_digests(chunk.content)
        if not valid_quote:
            issues.append(f"Quote citation {ref.evidence_id} is not a span of its canonical chunk.")
    return issues


def make_grounding_verification_node(
    grounding_llm: Runnable[list[Any], GroundingVerificationOutput],
) -> Callable[[LegalAgentState, RunnableConfig | None], Awaitable[dict[str, Any]]]:
    async def grounding_verification_node(
        state: LegalAgentState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        log = logger.bind(node="grounding_verification")

        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}

        summary_parts: list[str] = []
        risk_analysis = state.get("risk_analysis")
        if risk_analysis:
            summary_parts.append(
                "RISK ANALYSIS:\n"
                + json.dumps(risk_analysis.model_dump(mode="json"), ensure_ascii=False)
            )
        compliance_result = state.get("compliance_result")
        if compliance_result:
            summary_parts.append(
                "COMPLIANCE ANALYSIS:\n"
                + json.dumps(compliance_result.model_dump(mode="json"), ensure_ascii=False)
            )
        quote_issues = await _verify_quotes_against_canonical_source(state, config)

        try:
            raw_result = await _invoke_structured(
                grounding_llm,
                _GROUNDING_SYSTEM_PROMPT,
                "\n\n".join(summary_parts),
            )
        except (OutputParserException, StructuredOutputValidationError, ValidationError) as exc:
            exc.add_note(f"node=grounding_verification, doc_id={state['doc_id']}")
            log.bind(error=str(exc)).exception("grounding_output_invalid")
            result = _not_assessable_grounding_output(
                "Grounding was not assessable because the verifier returned invalid structured output.",
                _structural_grounding_issues(state),
            )
            errors = [
                _analysis_failure(
                    "grounding_verification",
                    "INVALID_GROUNDING_OUTPUT",
                    "The grounding verifier returned no validated structured result.",
                )
            ]
        except Exception as exc:
            exc.add_note(f"node=grounding_verification, doc_id={state['doc_id']}")
            log.exception("grounding_verifier_unavailable")
            result = _not_assessable_grounding_output(
                "Grounding was not assessable because the verifier was unavailable.",
                _structural_grounding_issues(state),
            )
            errors = [
                _analysis_failure(
                    "grounding_verification",
                    "GROUNDING_VERIFIER_UNAVAILABLE",
                    "Grounding could not be verified because the verifier was unavailable.",
                )
            ]
        else:
            ledger, blocked = _charge_model_result(ledger, raw_result)
            if blocked is not None:
                return {**blocked, "budget_ledger": ledger}
            parsed_result = _agent_structured_response(raw_result, GroundingVerificationOutput)
            if parsed_result is None:
                result = _not_assessable_grounding_output(
                    "Grounding was not assessable because the verifier returned no validated structured output.",
                    _structural_grounding_issues(state),
                )
                errors = [
                    _analysis_failure(
                        "grounding_verification",
                        "INVALID_GROUNDING_OUTPUT",
                        "The grounding verifier returned no validated structured result.",
                    )
                ]
            else:
                result = _apply_grounding_gate(
                    state, cast("GroundingVerificationOutput", parsed_result)
                )
                if quote_issues:
                    result = result.model_copy(
                        update={
                            "verified": False,
                            "status": GroundingStatus.NOT_ASSESSABLE,
                            "unverified_claims": [
                                *result.unverified_claims,
                                *quote_issues,
                            ],
                        }
                    )
                errors = []
                if result.status != GroundingStatus.VERIFIED:
                    errors.append(
                        _analysis_failure(
                            "grounding_verification",
                            "GROUNDING_GATE_REJECTED",
                            "One or more analysis claims could not pass the mandatory grounding gate.",
                        )
                    )
        log.info("grounding_verified", verified=result.verified, status=result.status)

        return {
            "grounding": result,
            "risk_analysis": state.get("risk_analysis"),
            "compliance_result": state.get("compliance_result"),
            "claim_ledger": list(state.get("claim_ledger", [])),
            "evidence_policy": state.get("evidence_policy", "mechanically_verified"),
            "coverage": dict(state.get("coverage", {})),
            "status": WorkflowStatus.AWAITING_HUMAN_REVIEW,
            "errors": errors,
            "budget_ledger": ledger,
        }

    return grounding_verification_node


def _structural_grounding_issues(state: LegalAgentState) -> list[str]:
    """Return deterministic grounding failures before trusting an LLM verdict."""
    issues: list[str] = []
    coverage = state.get("coverage", {})
    if coverage.get("documents_failed", 0) or coverage.get("documents_not_ready", 0):
        issues.append(
            "Evidence coverage is partial; failed or not-ready documents prevent a complete assessment."
        )
    risk_analysis = state.get("risk_analysis")
    if risk_analysis is None or risk_analysis.assessment_status != AssessmentStatus.ASSESSED:
        issues.append(
            "Risk analysis is not assessed; no risk conclusion may be presented as verified."
        )
    elif risk_analysis.overall_label is None:
        issues.append(
            "Risk analysis has no overall label; no risk conclusion may be presented as verified."
        )
    compliance_result = state.get("compliance_result")
    if (
        compliance_result is None
        or compliance_result.compliance_status == ComplianceStatus.NOT_ASSESSABLE
    ):
        issues.append(
            "Compliance is not assessable; no compliance conclusion may be presented as compliant."
        )
    elif compliance_result.compliance_status == ComplianceStatus.COMPLIANT and any(
        finding.insufficient_basis for finding in compliance_result.findings
    ):
        issues.append(
            "Compliance includes a finding with insufficient basis; it cannot be presented as compliant."
        )

    for finding in risk_analysis.findings if risk_analysis else []:
        if not finding.citations:
            issues.append(f"Risk finding {finding.risk_id} has no citation evidence.")
        if any(
            not citation.claim.strip() or not citation.source.strip() or citation.confidence <= 0.0
            for citation in finding.citations
        ):
            issues.append(f"Risk finding {finding.risk_id} has an incomplete citation.")
        issues.extend(_finding_evidence_issues(state, finding.citations, finding.risk_id))
    for finding in compliance_result.findings if compliance_result else []:
        if not finding.citations:
            issues.append(f"Compliance finding {finding.finding_id} has no citation evidence.")
        if any(
            not citation.claim.strip() or not citation.source.strip() or citation.confidence <= 0.0
            for citation in finding.citations
        ):
            issues.append(f"Compliance finding {finding.finding_id} has an incomplete citation.")
        issues.extend(_finding_evidence_issues(state, finding.citations, finding.finding_id))
    return issues


def _finding_evidence_issues(  # noqa: PLR0912 — closes handle, scope, quote, and policy gates
    state: LegalAgentState,
    citations: list[Any],
    finding_id: str,
) -> list[str]:
    """Apply the mechanical evidence-handle checks before model entailment."""

    registry = state.get("evidence_registry", {})
    allowed = set(state.get("allowed_doc_ids", ())) or {state.get("doc_id", "")}
    issues: list[str] = []
    for citation in citations:
        evidence_id = citation.evidence_id or (
            citation.source if citation.source in registry else None
        )
        if not evidence_id:
            issues.append(f"Finding {finding_id} cites no run-issued evidence handle.")
            continue
        ref = registry.get(evidence_id)
        if ref is None:
            issues.append(f"Finding {finding_id} cites an unknown or forged evidence handle.")
            continue
        if citation.source not in {evidence_id, ref.chunk_id}:
            issues.append(f"Finding {finding_id} carries a forged citation source.")
        if ref.document_id not in allowed:
            issues.append(f"Finding {finding_id} cites evidence outside the authorized scope.")
        if citation.contract_index is not None and citation.contract_index != ref.contract_index:
            issues.append(f"Finding {finding_id} cites a different contract identity.")
        if (
            state.get("evidence_policy", "mechanically_verified") == "graph_indexed_only"
            and not ref.graph_indexed
        ):
            issues.append(f"Finding {finding_id} cites evidence that is not Graphiti indexed.")
        expected_run = state.get("run_id") or state.get("correlation_id")
        if ref.run_id is not None and ref.run_id != expected_run:
            issues.append(f"Finding {finding_id} cites evidence from another run.")
        if ref.thread_id is not None and ref.thread_id != state.get("thread_id"):
            issues.append(f"Finding {finding_id} cites evidence from another thread.")
        if ref.checkpoint_id != _checkpoint_scope(state):
            issues.append(f"Finding {finding_id} cites evidence from another checkpoint.")
        if citation.quoted_text is not None:
            if ref.quote_digest is None:
                issues.append(f"Finding {finding_id} has a quote without a read-evidence digest.")
            elif (
                hashlib.sha256(citation.quoted_text.encode()).hexdigest() != ref.quote_digest
                and hashlib.sha256(" ".join(citation.quoted_text.split()).encode()).hexdigest()
                not in ref.quote_digests
            ):
                issues.append(
                    f"Finding {finding_id} has a quote that does not match read evidence."
                )
    return issues


def _downgrade_findings_and_build_ledger(
    state: LegalAgentState,
) -> list[ClaimLedgerEntry]:
    """Attach evidence metadata and downgrade unsupported material findings."""

    registry = state.get("evidence_registry", {})
    ledger: list[ClaimLedgerEntry] = []
    risk = state.get("risk_analysis")
    if risk is not None:
        findings = []
        for finding in risk.findings:
            ids = [
                citation.evidence_id or citation.source
                for citation in finding.citations
                if (citation.evidence_id or citation.source) in registry
            ]
            refs = [registry[evidence_id] for evidence_id in ids]
            supported = not _finding_evidence_issues(state, finding.citations, finding.risk_id)
            status = (
                ClaimSupportStatus.SUPPORTED if supported else ClaimSupportStatus.NOT_ASSESSABLE
            )
            findings.append(
                finding.model_copy(
                    update={
                        "evidence_refs": refs,
                        "support_status": status,
                        "contract_index": (
                            finding.contract_index
                            if finding.contract_index is not None
                            else (refs[0].contract_index if len(refs) == 1 else None)
                        ),
                    }
                )
            )
            ledger.append(
                ClaimLedgerEntry(
                    claim_id=finding.risk_id,
                    claim=finding.title,
                    evidence_ids=ids,
                    contract_index=(refs[0].contract_index if len(refs) == 1 else None),
                    support_status=status,
                    notes=None if supported else "Mechanical evidence gate rejected this finding.",
                )
            )
        state["risk_analysis"] = risk.model_copy(update={"findings": findings})
        if any(item.support_status != ClaimSupportStatus.SUPPORTED for item in findings):
            state["risk_analysis"] = state["risk_analysis"].model_copy(
                update={
                    "assessment_status": AssessmentStatus.NOT_ASSESSABLE,
                    "overall_label": None,
                }
            )

    compliance = state.get("compliance_result")
    if compliance is not None:
        findings = []
        for finding in compliance.findings:
            ids = [
                citation.evidence_id or citation.source
                for citation in finding.citations
                if (citation.evidence_id or citation.source) in registry
            ]
            refs = [registry[evidence_id] for evidence_id in ids]
            supported = not _finding_evidence_issues(state, finding.citations, finding.finding_id)
            status = (
                ClaimSupportStatus.SUPPORTED if supported else ClaimSupportStatus.NOT_ASSESSABLE
            )
            findings.append(
                finding.model_copy(
                    update={
                        "evidence_refs": refs,
                        "support_status": status,
                        "insufficient_basis": finding.insufficient_basis or not supported,
                        "contract_index": (
                            finding.contract_index
                            if finding.contract_index is not None
                            else (refs[0].contract_index if len(refs) == 1 else None)
                        ),
                    }
                )
            )
            ledger.append(
                ClaimLedgerEntry(
                    claim_id=finding.finding_id,
                    claim=finding.explanation,
                    evidence_ids=ids,
                    contract_index=(refs[0].contract_index if len(refs) == 1 else None),
                    support_status=status,
                    notes=None if supported else "Mechanical evidence gate rejected this finding.",
                )
            )
        state["compliance_result"] = compliance.model_copy(update={"findings": findings})
        if any(item.support_status != ClaimSupportStatus.SUPPORTED for item in findings):
            state["compliance_result"] = state["compliance_result"].model_copy(
                update={
                    "compliance_status": ComplianceStatus.NOT_ASSESSABLE,
                    "overall_compliant": None,
                }
            )
    return ledger


def _apply_grounding_gate(
    state: LegalAgentState,
    result: GroundingVerificationOutput,
) -> GroundingVerificationOutput:
    structural_issues = _structural_grounding_issues(state)
    ledger = _downgrade_findings_and_build_ledger(state)
    if ledger:
        state["claim_ledger"] = ledger
    if result.unverified_claims:
        structural_issues.append(
            "The grounding verifier reported unsupported claims; the result cannot be verified."
        )
    if not structural_issues and result.verified:
        refs = [
            state["evidence_registry"][citation.evidence_id or citation.source]
            for citation in result.citations
            if (citation.evidence_id or citation.source) in state.get("evidence_registry", {})
        ]
        return result.model_copy(
            update={
                "status": GroundingStatus.VERIFIED,
                "verified": True,
                "evidence_refs": refs,
            }
        )

    claims = list(result.unverified_claims)
    for issue in structural_issues:
        if issue not in claims:
            claims.append(issue)
    if not claims:
        claims.append("The grounding verifier did not establish support for the analysis.")

    status = (
        GroundingStatus.NOT_ASSESSABLE
        if any("not assessed" in claim or "not assessable" in claim for claim in claims)
        else GroundingStatus.UNVERIFIED
    )
    return result.model_copy(
        update={
            "verified": False,
            "status": status,
            "unverified_claims": claims,
            "notes": result.notes
            + " Mandatory grounding gate rejected at least one structural condition.",
        }
    )


def _not_assessable_grounding_output(
    reason: str,
    structural_issues: list[str],
) -> GroundingVerificationOutput:
    claims = list(structural_issues)
    if reason not in claims:
        claims.insert(0, reason)
    return GroundingVerificationOutput(
        verified=False,
        status=GroundingStatus.NOT_ASSESSABLE,
        unverified_claims=claims,
        notes=reason,
        citations=[],
    )


def _mechanical_final_citations(
    state: LegalAgentState,
    report: FinalReport,
) -> list[Citation]:
    """Derive final citations from validated finding citations only."""
    registry = state.get("evidence_registry", {})
    # Legacy checkpoints predate server-issued evidence handles. Preserve
    # their already-validated report citations until they are re-run through
    # the current evidence pipeline; current runs always take the strict path.
    if not registry:
        return list(report.citations)
    candidates = [
        citation
        for finding in (*report.risk_findings, *report.compliance_findings)
        for citation in finding.citations
    ]
    grounding = state.get("grounding")
    if grounding is not None:
        candidates.extend(grounding.citations)
    retained: list[Citation] = []
    seen: set[tuple[str, str, str | None]] = set()
    for citation in candidates:
        evidence_id = citation.evidence_id
        if evidence_id is None:
            continue
        if _finding_evidence_issues(state, [citation], "final_report"):
            continue
        ref = registry.get(evidence_id)
        retained_citation = citation
        if ref is not None and citation.contract_index is None:
            retained_citation = citation.model_copy(update={"contract_index": ref.contract_index})
        key = (evidence_id, retained_citation.claim, retained_citation.quoted_text)
        if key in seen:
            continue
        seen.add(key)
        retained.append(retained_citation)
    return retained


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
            "grounding_status": grounding.status.value if grounding and grounding.status else None,
            "grounding_verified": grounding.verified if grounding else False,
            "evidence_policy": state.get("evidence_policy", "mechanically_verified"),
            "coverage": dict(state.get("coverage", {})),
            "evidence_refs": [
                ref.model_dump(mode="json") for ref in state.get("evidence_registry", {}).values()
            ],
            "compliance_status": (
                compliance_result.compliance_status.value if compliance_result else "not_assessable"
            ),
            "segments": [
                seg.model_dump() for seg in state["segments"][:_HUMAN_REVIEW_SEGMENT_PREVIEW]
            ],
            "message": "Please review findings, add overrides if needed, and approve to finalize",
        }

        human_response: dict[str, Any] = interrupt(review_payload)

        if human_response.get("action") == "reject":
            log.warning("human_review_rejected")
            return _fail(
                "human_review",
                "REVIEW_REJECTED",
                human_response.get("feedback") or "Rejected at human review",
                retryable=True,
            )

        raw_overrides = human_response.get("overrides") or []
        overrides = [ReviewOverride.model_validate(o) for o in raw_overrides]

        review_output = HumanReviewOutput(
            reviewer_id=human_response.get("reviewer_id") or "unknown",
            reviewer_role=human_response.get("reviewer_role") or "reviewer",
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
            "self_reviewed": review_output.reviewer_id == state.get("user_id"),
            "status": WorkflowStatus.FINALIZING,
        }

    return human_review_node


def make_finalization_node(
    finalization_llm: Runnable[list[Any], FinalReport],
) -> StateNode:
    async def finalization_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(node="finalization", doc_id=state["doc_id"])
        ledger, blocked = _budget_gate(state, BudgetDimension.MODEL_CALLS)
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        risk_analysis = state.get("risk_analysis")
        compliance_result = state.get("compliance_result")
        human_review = state.get("human_review")
        grounding = state.get("grounding")

        if grounding is None or not grounding.verified:
            log.warning(
                "finalization_blocked_by_grounding",
                grounding_status=grounding.status if grounding else None,
            )
            return _fail(
                "finalization",
                "GROUNDING_NOT_VERIFIED",
                "Finalization is blocked until all material findings pass grounding verification.",
                retryable=True,
            )
        if (
            compliance_result is None
            or compliance_result.compliance_status == ComplianceStatus.NOT_ASSESSABLE
        ):
            log.warning("finalization_blocked_by_compliance_status")
            return _fail(
                "finalization",
                "COMPLIANCE_NOT_ASSESSABLE",
                "Finalization is blocked because compliance could not be assessed.",
                retryable=True,
            )
        if human_review is None or not human_review.approved:
            log.warning("finalization_blocked_by_human_review")
            return _fail(
                "finalization",
                "HUMAN_REVIEW_NOT_APPROVED",
                "Finalization is blocked until an authorized human reviewer approves the analysis.",
                retryable=True,
            )

        context_parts: list[str] = [
            f"Document ID: {state['doc_id']}",
            f"Allowed document IDs: {', '.join(state.get('allowed_doc_ids', [state['doc_id']]))}",
            f"User query: {state['user_query']}",
            f"Grounding status: {grounding.status}",
            f"Compliance status: {compliance_result.compliance_status}",
        ]
        if risk_analysis:
            context_parts.append(
                "RISK ANALYSIS (authoritative state):\n"
                + json.dumps(risk_analysis.model_dump(mode="json"), ensure_ascii=False)
            )
        context_parts.append(
            "CANONICAL SEGMENTS (identity must be retained in citations):\n"
            + json.dumps(
                [
                    {
                        "clause_id": segment.clause_id,
                        "document_id": segment.document_id or state["doc_id"],
                        "document_version": segment.document_version,
                        "contract_index": segment.contract_index,
                        "chunk_id": segment.chunk_id,
                        "chunk_index": segment.chunk_index,
                        "page_no": segment.page_no,
                        "text": segment.text[:_CLAUSE_CONTEXT_CHARS],
                    }
                    for segment in state["segments"]
                ],
                ensure_ascii=False,
            )
        )
        if compliance_result:
            context_parts.append(
                "COMPLIANCE ANALYSIS (authoritative state):\n"
                + json.dumps(compliance_result.model_dump(mode="json"), ensure_ascii=False)
            )
        context_parts.append(
            "GROUNDING VERIFICATION (authoritative state):\n"
            + json.dumps(grounding.model_dump(mode="json"), ensure_ascii=False)
        )
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
        try:
            report_result = await finalization_llm.ainvoke(messages)
            ledger, blocked = _charge_model_result(ledger, report_result)
            if blocked is not None:
                return {**blocked, "budget_ledger": ledger}
            report = (
                report_result
                if isinstance(report_result, FinalReport)
                else FinalReport.model_validate(report_result)
            )
        except (OutputParserException, ValidationError) as exc:
            exc.add_note(f"node=finalization, doc_id={state['doc_id']}")
            log.bind(error=str(exc)).exception("finalization_output_invalid")
            return _fail(
                "finalization",
                "INVALID_FINAL_REPORT",
                "The final report did not meet the required evidence schema.",
                retryable=True,
            )
        except Exception as exc:
            exc.add_note(f"node=finalization, doc_id={state['doc_id']}")
            log.exception("finalization_agent_unavailable")
            return _fail(
                "finalization",
                "FINALIZATION_AGENT_UNAVAILABLE",
                "The final report could not be produced because the finalizer was unavailable.",
                retryable=True,
            )
        report = report.model_copy(
            update={
                "risk_findings": list(risk_analysis.findings) if risk_analysis else [],
                "compliance_findings": list(compliance_result.findings),
                "human_overrides": list(human_review.overrides) if human_review else [],
                "compliance_status": compliance_result.compliance_status,
                "grounding_status": GroundingStatus.VERIFIED,
                "evidence_policy": state.get("evidence_policy", "mechanically_verified"),
                "coverage": dict(state.get("coverage", {})),
                "self_reviewed": bool(state.get("self_reviewed", False)),
                "reviewer_id": getattr(human_review, "reviewer_id", None),
                "reviewer_role": getattr(human_review, "reviewer_role", None),
                "contract_indices": sorted(
                    {
                        ref.contract_index
                        for ref in state.get("evidence_registry", {}).values()
                        if ref.contract_index is not None
                    }
                ),
            }
        )
        final_citations = _mechanical_final_citations(state, report)
        if not final_citations:
            log.warning("finalization_blocked_by_invalid_citations")
            return _fail(
                "finalization",
                "FINAL_CITATIONS_INVALID",
                "The final report did not contain mechanically valid evidence citations.",
                retryable=True,
            )
        report = report.model_copy(update={"citations": final_citations})
        log.info("finalization_completed", suggested_actions=len(report.suggested_actions))

        return {
            "final_report": report,
            "budget_ledger": ledger,
            "status": WorkflowStatus.PERSISTING_MEMORY,
        }

    return finalization_node


def make_persist_memory_node(memory_service: Any) -> StateNode:
    """Persist the approved final report through the agent-memory service.

    Gated on human approval (7.1): an unapproved run performs zero memory writes
    at any trust level and still completes. Fail-open by contract (7.2): a
    memory failure records ``COGNEE_WRITE_FAILED`` and never aborts a completed
    legal analysis. The document knowledge graph is not touched here at all
    (7.3) — agent memory and the entity graph are different stores with
    different owners.
    """

    async def persist_memory_node(state: LegalAgentState) -> dict[str, Any]:
        log = logger.bind(
            node="persist_memory",
            user_id=state["user_id"],
            doc_id=state["doc_id"],
        )

        long_term_refs: list[str] = list(state.get("long_term_refs", []))
        review = state.get("human_review")
        approved = bool(review and review.approved)

        final_report = state.get("final_report")
        if final_report is None or not approved:
            log.info("persist_memory_skipped", approved=approved)
            return {
                "long_term_refs": long_term_refs,
                "status": (
                    WorkflowStatus.FAILED
                    if state.get("status") == WorkflowStatus.FAILED
                    else WorkflowStatus.COMPLETED
                ),
            }

        # Pre-tenant checkpoints carry only user_id.  Treat that identity as
        # the personal tenant namespace during migration; current runs always
        # set the server-authorized tenant_id explicitly.
        tenant_id = str(state.get("tenant_id") or state.get("user_id") or "")
        if not tenant_id:
            # A report without a durable tenant boundary must never enter the
            # long-term store: the previous user-id namespace was not a
            # tenant isolation guarantee for shared workspaces.
            return {
                **_fail(
                    "persist_memory",
                    "TENANT_SCOPE_MISSING",
                    "Cannot persist a report without a tenant scope",
                    status=WorkflowStatus.COMPLETED,
                    retryable=False,
                ),
                "long_term_refs": long_term_refs,
            }

        try:
            report_text = str(final_report.model_dump_json())
            await memory_service.store_report(
                report_text=report_text,
                conversation_id=state["thread_id"],
                tenant_id=tenant_id,
            )
            ref_key = f"legal::{tenant_id}::reports#{state['doc_id']}"
            long_term_refs.append(ref_key)
            log.info("persist_memory_completed", ref_count=len(long_term_refs))

        except Exception as exc:  # noqa: BLE001 — fail-open by contract (7.2)
            exc.add_note("node=persist_memory")
            log.warning("persist_memory_failed", error=str(exc))
            return {
                **_fail(
                    "persist_memory",
                    "COGNEE_WRITE_FAILED",
                    str(exc),
                    status=WorkflowStatus.COMPLETED,
                    retryable=True,
                ),
                "long_term_refs": long_term_refs,
            }

        return {
            "long_term_refs": long_term_refs,
            "status": (
                WorkflowStatus.FAILED
                if state.get("status") == WorkflowStatus.FAILED
                else WorkflowStatus.COMPLETED
            ),
        }

    return persist_memory_node


def route_deep_research(
    state: LegalAgentState,
) -> Literal["deep_research", "orchestrator", "planner_generate"]:
    if state.get("status") == WorkflowStatus.PLAN_REJECTED:
        return "planner_generate"

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    for step in plan[current_step:]:
        if step.action == PlanActionType.SEARCH_PRECEDENTS:
            return "deep_research"
    return "orchestrator"


async def _run_scoped_research_step(
    evidence_service: Any | None,
    state: LegalAgentState,
    step: PlanStep,
    log: Any,
) -> tuple[EvidenceSearchResponse | None, list[AgentError]]:
    """Execute one research query through the server-scoped evidence seam."""
    if evidence_service is None:
        return None, [
            _analysis_failure(
                "deep_research",
                "DEEP_RESEARCH_SERVICE_UNAVAILABLE",
                "Precedent research was not assessable because no scoped evidence service was configured.",
            )
        ]

    user_id = state.get("user_id", "")
    tenant_id = state.get("tenant_id", "")
    allowed_doc_ids = tuple(dict.fromkeys(state.get("allowed_doc_ids", ())))
    evidence_policy = state.get("evidence_policy", "mechanically_verified")
    if evidence_policy not in {"mechanically_verified", "graph_indexed_only"}:
        evidence_policy = "mechanically_verified"
    if not user_id or not tenant_id or not allowed_doc_ids:
        return (
            EvidenceSearchResponse(
                status="unauthorized",
                evidence_policy=cast("EvidencePolicy", evidence_policy),
                message="Deep research has no complete server-owned authorization scope.",
            ),
            [],
        )

    thread_id = state.get("thread_id", "")
    try:
        response = await evidence_service.search_evidence(
            principal_id=user_id,
            tenant_id=tenant_id,
            allowed_doc_ids=allowed_doc_ids,
            thread_id=thread_id,
            run_id=state.get("run_id") or state.get("correlation_id") or thread_id,
            checkpoint_id=state.get("evidence_checkpoint_id"),
            request=EvidenceSearchRequest(query=step.description, limit=5),
            evidence_policy=cast("EvidencePolicy", evidence_policy),
        )
    except Exception as exc:
        exc.add_note(f"node=deep_research, step_id={step.step_id}")
        log.exception("deep_research_evidence_service_failed", step_id=step.step_id)
        return None, [
            _analysis_failure(
                "deep_research",
                "DEEP_RESEARCH_SERVICE_UNAVAILABLE",
                "Precedent research was not assessable because the evidence service was unavailable.",
            )
        ]

    if isinstance(response, EvidenceSearchResponse):
        return response, []
    try:
        return EvidenceSearchResponse.model_validate(response), []
    except ValidationError as exc:
        exc.add_note(f"node=deep_research, step_id={step.step_id}")
        return None, [
            _analysis_failure(
                "deep_research",
                "DEEP_RESEARCH_INVALID_RESPONSE",
                "Precedent research returned an invalid evidence envelope.",
            )
        ]


def _collect_scoped_research_response(
    response: EvidenceSearchResponse,
    step: PlanStep,
) -> tuple[dict[str, EvidenceRef], dict[str, int], list[AgentError]]:
    """Reduce a typed search response to checkpoint-safe metadata."""
    coverage: dict[str, int] = {
        key: value
        for key, value in response.coverage.model_dump(mode="json").items()
        if isinstance(value, int)
    }
    errors: list[AgentError] = []
    refs: dict[str, EvidenceRef] = {}
    if response.status == "ok":
        for item in response.items:
            ref = EvidenceRef.model_validate(item.evidence.model_dump(mode="json"))
            refs[ref.evidence_id] = ref
        if response.coverage.documents_ready < response.coverage.documents_requested:
            errors.append(
                _analysis_failure(
                    "deep_research",
                    "DEEP_RESEARCH_PARTIAL_COVERAGE",
                    f"Precedent research step {step.step_id} returned only partial document coverage.",
                )
            )
    else:
        errors.append(
            _analysis_failure(
                "deep_research",
                f"DEEP_RESEARCH_{response.status.upper()}",
                response.message
                or f"Precedent research step {step.step_id} was not assessable ({response.status}).",
            )
        )
    return refs, coverage, errors


def make_deep_research_node(
    deep_research_tool: BaseTool | None = None,
    *,
    evidence_service: Any | None = None,
) -> StateNode:
    """Run only the precedent-search steps in an approved plan.

    Deep research is deliberately an adapter over the scoped evidence service,
    not a second model/tool contract.  The service receives the server-owned
    tenant and document allowlist from the outer Saul state and returns typed
    evidence handles.  Search snippets and source content never enter graph
    state; only ``EvidenceRef`` metadata, coverage, and typed errors do.

    ``deep_research_tool`` remains as a compatibility parameter for old test
    factories, but is intentionally not invoked.  The former tool accepted a
    mismatched ``question`` argument and returned hydrated precedent text,
    which could both fail at runtime and leak source content into checkpoints.
    """

    # Keep the old parameter source-compatible for external graph factories;
    # it is deliberately not a fallback because it cannot issue scoped handles.
    del deep_research_tool

    async def deep_research_node(state: LegalAgentState) -> dict[str, Any]:  # noqa: PLR0914 — bounded fan-out bookkeeping
        log = logger.bind(
            node="deep_research",
            user_id=state["user_id"],
            thread_id=state["thread_id"],
        )

        plan = state.get("plan", [])
        current_step = state.get("current_step", 0)

        indexed_research_steps = [
            (index, step)
            for index, step in enumerate(plan[current_step:], start=current_step)
            if step.action == PlanActionType.SEARCH_PRECEDENTS
        ]

        if not indexed_research_steps:
            return {}

        research_steps = [step for _, step in indexed_research_steps]

        ledger, blocked = _budget_gate(
            state, BudgetDimension.DEEP_RESEARCH_FANOUT, len(research_steps)
        )
        if blocked is not None:
            return {**blocked, "budget_ledger": ledger}
        retrieval_ledger, retrieval_blocked = _budget_gate(
            {**state, "budget_ledger": ledger},
            BudgetDimension.RETRIEVAL_CALLS,
            len(research_steps),
        )
        if retrieval_blocked is not None:
            return {**retrieval_blocked, "budget_ledger": retrieval_ledger}
        ledger = retrieval_ledger

        log.info(
            "deep_research_started",
            step_count=len(research_steps),
            current_step=current_step,
        )

        refs: dict[str, EvidenceRef] = {}
        coverage: dict[str, int] = dict(state.get("coverage", {}))
        errors: list[AgentError] = []

        # A plan normally contains only a small number of research steps. Keep
        # the fan-out bounded by the planner's maximum rather than introducing
        # an unbounded recursive search loop.
        responses = await asyncio.gather(
            *(
                _run_scoped_research_step(evidence_service, state, step, log)
                for step in research_steps
            )
        )
        for step, (response, step_errors) in zip(research_steps, responses, strict=True):
            errors.extend(step_errors)
            if response is None:
                continue
            response_refs, response_coverage, response_errors = _collect_scoped_research_response(
                response, step
            )
            refs.update(response_refs)
            errors.extend(response_errors)
            for key, value in response_coverage.items():
                coverage[key] = max(coverage.get(key, 0), value)

        merged_registry = dict(state.get("evidence_registry", {}))
        merged_registry.update(refs)
        # The legacy field is retained for checkpoint compatibility, but now
        # contains a compact JSON metadata list rather than research text.
        metadata_only = json.dumps(
            [ref.model_dump(mode="json") for ref in refs.values()],
            ensure_ascii=False,
            sort_keys=True,
        )
        current_status = state.get("status", WorkflowStatus.PLAN_APPROVED)
        if not refs and errors:
            current_status = WorkflowStatus.FAILED
        return {
            "deep_research_results": metadata_only,
            "evidence_registry": merged_registry,
            "coverage": coverage,
            "errors": errors,
            "status": current_status,
            "current_step": indexed_research_steps[-1][0] + 1,
            "budget_ledger": ledger,
        }

    return deep_research_node
