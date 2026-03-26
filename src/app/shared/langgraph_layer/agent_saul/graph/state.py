"""
LegalAgentState: the single source of truth for Agent Saul.

Design rules:
- All nodes are pure functions over this state.
- No hidden memory. No implicit mutation.
- Reducers declared via Annotated — last-write-wins for scalars,
  operator.add for append-only accumulation lists.
- schema_version guards the AsyncPostgresSaver hydration node against
  breaking replays when the TypedDict evolves.
"""

import operator
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkflowStatus(StrEnum):
    INITIALIZED = "initialized"
    QNA_CLARIFICATION = "qna_clarification"
    PLAN_PENDING = "plan_pending"
    PLAN_AWAITING_APPROVAL = "plan_awaiting_approval"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"
    INGESTING = "ingesting"
    NORMALIZING = "normalizing"
    SEGMENTING = "segmenting"
    EXTRACTING_ENTITIES = "extracting_entities"
    MAPPING_RELATIONSHIPS = "mapping_relationships"
    ANALYZING_RISKS = "analyzing_risks"
    CHECKING_COMPLIANCE = "checking_compliance"
    VERIFYING_GROUNDING = "verifying_grounding"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    FINALIZING = "finalizing"
    PERSISTING_MEMORY = "persisting_memory"
    COMPLETED = "completed"
    FAILED = "failed"


class HITLInterruptType(StrEnum):
    CLARIFICATION_NEEDED = "clarification_needed"
    PLAN_APPROVAL = "plan_approval"
    OCR_REUPLOAD = "ocr_reupload"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class PlanActionType(StrEnum):
    SEARCH_PRECEDENTS = "search_precedents"
    EXTRACT_CLAUSES = "extract_clauses"
    RISK_ANALYSIS = "risk_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    SUMMARIZE = "summarize"


class OrchestratorActionType(StrEnum):
    START_PIPELINE = "start_pipeline"
    CONTINUE = "continue"
    SYNTHESIZE = "synthesize"
    DONE = "done"


class RiskLabel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClauseType(StrEnum):
    INDEMNITY = "indemnity"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    ARBITRATION = "arbitration"
    TERMINATION = "termination"
    GOVERNING_LAW = "governing_law"
    CONFIDENTIALITY = "confidentiality"
    PAYMENT = "payment"
    IP_OWNERSHIP = "ip_ownership"
    OTHER = "other"


class RelationshipType(StrEnum):
    INDEMNIFIES = "indemnifies"
    TRIGGERED_BY = "triggered_by"
    OVERRIDDEN_BY = "overridden_by"
    DEADLINE = "deadline"
    RESTRICTS = "restricts"
    OBLIGES = "obliges"


# ---------------------------------------------------------------------------
# Citation enforcement — all LLM outputs that make claims MUST include this.
# ---------------------------------------------------------------------------


class Citation(BaseModel, frozen=True):
    claim: str = Field(description="The specific claim being made")
    source: str = Field(description="Document section, statute, or precedent ID")
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Planning schemas
# ---------------------------------------------------------------------------


class PlanStep(BaseModel, frozen=True):
    step_id: str
    action: PlanActionType
    description: str
    input_keys: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class OrchestratorAction(BaseModel, frozen=True):
    action_type: OrchestratorActionType
    target_node: str = Field(default="")
    reflection: str = Field(description="Orchestrator's reasoning for this action")
    retry_node: str | None = None


# ---------------------------------------------------------------------------
# Document processing schemas
# ---------------------------------------------------------------------------


class DocumentSection(BaseModel, frozen=True):
    section_id: str
    title: str
    content: str
    level: int
    parent_id: str | None = None
    clause_ref: str | None = Field(
        default=None,
        description="Normalized ref e.g. Clause 7.2(b) → resolved section_id",
    )


class NormalizedDocument(BaseModel, frozen=True):
    document_id: str
    sections: list[DocumentSection]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClauseSegment(BaseModel, frozen=True):
    clause_id: str
    clause_type: ClauseType
    text: str
    section_ref: str
    start_char: int
    end_char: int


# ---------------------------------------------------------------------------
# Entity extraction — schema-locked, no interpretation
# ---------------------------------------------------------------------------


class EntityType(StrEnum):
    PARTY = "PARTY"
    DATE = "DATE"
    MONEY = "MONEY"
    JURISDICTION = "JURISDICTION"
    OBLIGATION = "OBLIGATION"
    CONDITION = "CONDITION"


class CitedEntity(BaseModel, frozen=True):
    entity_id: str
    entity_type: EntityType
    value: str
    party: str | None = None
    clause_id: str
    citation: Citation


# ---------------------------------------------------------------------------
# Relationship mapping
# ---------------------------------------------------------------------------


class LegalRelationship(BaseModel, frozen=True):
    edge_id: str
    from_node: str
    relationship: RelationshipType
    to_node: str
    clause_id: str
    citation: Citation


# ---------------------------------------------------------------------------
# Risk + Compliance
# ---------------------------------------------------------------------------


class RiskFinding(BaseModel, frozen=True):
    risk_id: str
    clause_id: str
    label: RiskLabel
    title: str
    explanation: str
    citations: list[Citation]
    suggested_revision: str | None = None


class RiskAnalysisOutput(BaseModel, frozen=True):
    findings: list[RiskFinding]
    overall_label: RiskLabel
    summary: str


class ComplianceFinding(BaseModel, frozen=True):
    finding_id: str
    clause_id: str
    statute: str
    is_compliant: bool
    explanation: str
    citations: list[Citation]
    insufficient_basis: bool = Field(
        default=False,
        description="True when retrieved sources < confidence threshold",
    )


class ComplianceOutput(BaseModel, frozen=True):
    findings: list[ComplianceFinding]
    jurisdiction: str
    overall_compliant: bool
    summary: str


class GroundingVerificationOutput(BaseModel, frozen=True):
    verified: bool
    unverified_claims: list[str]
    notes: str


# ---------------------------------------------------------------------------
# Human review (MANDATORY — legal liability + model improvement)
# ---------------------------------------------------------------------------


class ReviewOverride(BaseModel):
    clause_id: str
    original_label: str
    override_label: str
    reason_code: str
    comment: str | None = None


class HumanReviewOutput(BaseModel):
    reviewer_id: str
    reviewer_role: str
    overrides: list[ReviewOverride] = Field(default_factory=list)
    approved: bool
    notes: str | None = None


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


class FinalReport(BaseModel, frozen=True):
    document_id: str
    summary: str
    risk_findings: list[RiskFinding]
    compliance_findings: list[ComplianceFinding]
    human_overrides: list[ReviewOverride]
    suggested_actions: list[str]
    citations: list[Citation]


# ---------------------------------------------------------------------------
# Error record
# ---------------------------------------------------------------------------


class AgentError(BaseModel, frozen=True):
    node: str
    code: str
    message: str
    retryable: bool = True


# ---------------------------------------------------------------------------
# Sub-state for Send fan-out (entity_extraction receives this per clause)
# ---------------------------------------------------------------------------


class ClauseExtractionInput(BaseModel, frozen=True):
    clause: ClauseSegment
    document_context: dict[str, Any]


# ---------------------------------------------------------------------------
# LegalAgentState
#
# Reducers:
#   messages           → add_messages   (deduplicates by ID, handles trim)
#   segments           → operator.add   (parallel Send fan-out accumulation)
#   extracted_entities → operator.add   (parallel fan-out result accumulation)
#   relationships      → operator.add   (parallel fan-out result accumulation)
#   errors             → operator.add   (append-only audit log)
#
# All other fields: last-write-wins (standard LangGraph default).
# ---------------------------------------------------------------------------


class LegalAgentState(TypedDict):
    # --- Identity ---
    user_id: str
    thread_id: str
    correlation_id: str
    schema_version: int  # bump when TypedDict evolves; guards hydration node

    # --- Document reference ---
    doc_id: str
    document_text: str | None  # populated by ingestion node

    # --- Conversation messages (LangGraph message reducer) ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Intent ---
    user_query: str
    qna_confidence: float  # 0.0-1.0; < 0.7 triggers HITL clarification

    # --- Planning ---
    plan: list[PlanStep]
    current_step: int
    plan_approved: bool
    orchestrator_action: OrchestratorAction | None

    # --- Processing pipeline outputs ---
    normalized_document: NormalizedDocument | None
    segments: Annotated[list[ClauseSegment], operator.add]
    extracted_entities: Annotated[list[CitedEntity], operator.add]
    relationships: Annotated[list[LegalRelationship], operator.add]

    # --- Analysis ---
    risk_analysis: RiskAnalysisOutput | None
    compliance_result: ComplianceOutput | None
    grounding: GroundingVerificationOutput | None

    # --- Human review ---
    human_review: HumanReviewOutput | None

    # --- Final ---
    final_report: FinalReport | None

    # --- Memory ---
    long_term_refs: list[str]  # cognee store namespace keys
    working_memory: dict[str, Any]  # ephemeral cross-node scratch space

    # --- Control ---
    status: WorkflowStatus
    errors: Annotated[list[AgentError], operator.add]
    retry_count: int
    permissions: dict[str, bool]


# Exported node names — used by service to filter astream_events
GRAPH_NODE_NAMES: frozenset[str] = frozenset(
    {
        "gateway",
        "qna",
        "orchestrator",
        "planner",
        "ingestion",
        "normalization",
        "segmentation",
        "entity_extraction",
        "relationship_mapping",
        "risk_analysis",
        "compliance",
        "grounding_verification",
        "human_review",
        "finalization",
        "persist_memory",
    }
)
