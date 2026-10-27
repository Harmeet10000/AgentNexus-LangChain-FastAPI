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

import hashlib
import operator
import re
from enum import StrEnum
from typing import Annotated, Any, Final, TypedDict, get_args, get_origin, get_type_hints

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator

from app.features.agent_saul.budget import BudgetLedger

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


class IngestionDocumentStatus(StrEnum):
    """Server-derived readiness exposed to the orchestrator.

    These are deliberately not inferred from retrieval result cardinality:
    an empty result for a ready document is different from a document that is
    still processing or failed ingestion.
    """

    READY = "ready"
    WARNINGS = "warnings"
    NOT_READY = "not_ready"
    FAILED = "failed"


class IngestionDocumentState(BaseModel, frozen=True):
    """Bounded, typed readiness metadata for one authorized document."""

    document_id: str
    status: IngestionDocumentStatus
    warning_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    retryable: bool = False


def initial_ingestion_state(
    document_ids: list[str] | tuple[str, ...],
) -> list[IngestionDocumentState]:
    """Return a complete fail-closed readiness snapshot for a document scope."""

    return [
        IngestionDocumentState(
            document_id=document_id,
            status=IngestionDocumentStatus.NOT_READY,
        )
        for document_id in dict.fromkeys(document_ids)
    ]


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


class AssessmentStatus(StrEnum):
    """Whether an analysis produced a usable assessment.

    ``NOT_ASSESSED`` means the worker could not produce an assessment (for
    example, its tool or structured-output path was unavailable).  It is
    intentionally distinct from a low-risk or compliant result: absence of
    analysis is not evidence of safety.
    """

    ASSESSED = "assessed"
    NOT_ASSESSED = "not_assessed"
    NOT_ASSESSABLE = "not_assessable"


class ComplianceStatus(StrEnum):
    """The only valid aggregate compliance conclusions."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSABLE = "not_assessable"


class GroundingStatus(StrEnum):
    """Outcome of the mandatory grounding gate."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    NOT_ASSESSABLE = "not_assessable"


class ClaimSupportStatus(StrEnum):
    """Claim-level support outcome retained in the Saul checkpoint."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    NOT_ASSESSABLE = "not_assessable"


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
    # New evidence contract fields are optional for legacy checkpoints and
    # provider responses. The grounding gate requires them for material
    # findings before a report can be finalized.
    evidence_id: str | None = None
    # Quote text is a runtime entailment input, not durable evidence.  The
    # server-issued EvidenceRef retains only digests/handles across pauses.
    quoted_text: str | None = Field(default=None, max_length=2_000, exclude=True)
    quote_digest: str | None = None
    contract_index: int | None = None

    @model_validator(mode="before")
    @classmethod
    def retain_quote_digest_only_for_checkpoints(cls, value: Any) -> Any:
        """Drop source text at serialization while retaining its verifier key."""

        if not isinstance(value, dict):
            return value
        payload = dict(value)
        quoted = payload.get("quoted_text")
        if isinstance(quoted, str) and quoted and not payload.get("quote_digest"):
            payload["quote_digest"] = hashlib.sha256(quoted.encode()).hexdigest()
        return payload


class EvidenceRef(BaseModel, frozen=True):
    """Metadata for a server-issued evidence handle.

    Deliberately excludes source content. Content is dereferenced through the
    document evidence service and must never be copied into checkpoint state.
    The shape mirrors ``documents.dto.EvidenceRef`` so node outputs can be
    passed through without a lossy translation.
    """

    evidence_id: str
    chunk_id: str
    document_id: str
    document_version: int
    contract_index: int | None = None
    pipeline_version: str | None = None
    page_no: int = 0
    graph_indexed: bool = False
    retrieval_method: str = ""
    score: float = 0.0
    rank: int = 0
    quote_digest: str | None = None
    quote_digests: tuple[str, ...] = ()
    run_id: str | None = None
    thread_id: str | None = None
    checkpoint_id: str = ""


class ClaimLedgerEntry(BaseModel, frozen=True):
    """Atomic claim provenance retained without raw tool artifacts."""

    claim_id: str
    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    contract_index: int | None = None
    support_status: ClaimSupportStatus = ClaimSupportStatus.NOT_ASSESSABLE
    notes: str | None = None


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
    level: int = Field(ge=1, description="1-based hierarchy depth of this section")
    parent_id: str | None = None
    clause_ref: str | None = Field(
        default=None,
        description="Normalized ref e.g. Clause 7.2(b) → resolved section_id",
    )

    @model_validator(mode="before")
    @classmethod
    def infer_level_from_numbered_section_id(cls, value: Any) -> Any:
        """Recover an omitted level only when a numeric id makes depth explicit.

        Gemini Flash-Lite has occasionally omitted this required structured
        field. Numbered identifiers such as ``3.1.2`` encode an unambiguous
        1-based depth; prose labels (``Schedule A``) do not, and remain invalid
        when the model omits ``level``.
        """
        if not isinstance(value, dict) or "level" in value:
            return value
        section_id = value.get("section_id")
        if not isinstance(section_id, str) or re.fullmatch(r"\d+(?:\.\d+)*", section_id) is None:
            return value
        return {**value, "level": section_id.count(".") + 1}


class NormalizedDocument(BaseModel, frozen=True):
    document_id: str
    sections: list[DocumentSection] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def discard_only_empty_section_artifacts(cls, value: Any) -> Any:
        """Drop empty model artifacts without relaxing real section validation.

        An exact empty object carries no document information and is sometimes
        emitted by structured-output models. Partially populated objects remain
        invalid; the normalizer must never invent their missing legal content.
        """
        if not isinstance(value, dict):
            return value
        sections = value.get("sections")
        if not isinstance(sections, list):
            return value
        return {**value, "sections": [section for section in sections if section != {}]}


class ClauseSegment(BaseModel, frozen=True):
    """A clause's runtime view and its checkpoint-safe evidence identity.

    ``text`` is intentionally runtime-only.  Canonical chunks are read under
    the current authorization context and may be used by nodes during the
    active super-step, but retaining source text in a LangGraph checkpoint
    would turn an opaque evidence reference into a durable legal-payload
    copy.  ``exclude=True`` is honored by LangGraph's JSON-plus serializer;
    the empty default lets an old checkpoint hydrate without inventing text.
    Consumers must fail closed when a resumed segment needs text and its
    source cannot be dereferenced.
    """

    clause_id: str
    clause_type: ClauseType
    section_ref: str
    start_char: int
    end_char: int
    text: str = Field(default="", exclude=True)
    # Canonical evidence identity.  Defaults preserve v3 checkpoints and the
    # legacy LLM output schema; canonical ingestion always fills every field.
    document_id: str = ""
    document_version: int = 1
    chunk_id: str = ""
    chunk_index: int = 0
    contract_index: int | None = None
    page_no: int = 0
    chunk_kind: str = "generic"
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_digest: str | None = Field(
        default=None,
        description="SHA-256 of source content; source text is never checkpointed",
    )
    retrieval_pointer: str = Field(
        default="",
        description="Opaque server-side pointer used for just-in-time dereference",
    )


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
    contract_index: int | None = None


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
    contract_index: int | None = None


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
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    contract_index: int | None = None
    support_status: ClaimSupportStatus = ClaimSupportStatus.NOT_ASSESSABLE

    @field_validator("citations")
    @classmethod
    def _citations_non_empty(cls, value: list[Citation]) -> list[Citation]:
        # D-9: an uncited assertion is the failure mode. An empty citation
        # list must fail validation, not flow downstream with a warning.
        if not value:
            msg = "a risk finding must carry at least one citation"
            raise ValueError(msg)
        return value


class RiskAnalysisOutput(BaseModel, frozen=True):
    findings: list[RiskFinding]
    overall_label: RiskLabel | None = None
    summary: str
    assessment_status: AssessmentStatus = AssessmentStatus.ASSESSED


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
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    contract_index: int | None = None
    support_status: ClaimSupportStatus = ClaimSupportStatus.NOT_ASSESSABLE

    @field_validator("citations")
    @classmethod
    def _citations_non_empty(cls, value: list[Citation]) -> list[Citation]:
        # D-9: an uncited assertion is the failure mode. An empty citation
        # list must fail validation, not flow downstream with a warning.
        if not value:
            msg = "a compliance finding must carry at least one citation"
            raise ValueError(msg)
        return value


class ComplianceOutput(BaseModel, frozen=True):
    findings: list[ComplianceFinding]
    jurisdiction: str
    overall_compliant: bool | None = None
    summary: str
    compliance_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSABLE

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_status(cls, value: Any) -> Any:
        """Map the legacy boolean field to the explicit tri-state status.

        Existing persisted/model-produced payloads only contain
        ``overall_compliant``.  Preserve those payloads while ensuring a
        missing determination can never be interpreted as ``True``.
        """
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "compliance_status" not in payload:
            overall = payload.get("overall_compliant")
            payload["compliance_status"] = (
                ComplianceStatus.COMPLIANT
                if overall is True
                else ComplianceStatus.NON_COMPLIANT
                if overall is False
                else ComplianceStatus.NOT_ASSESSABLE
            )
        elif isinstance(payload["compliance_status"], str):
            payload["compliance_status"] = ComplianceStatus(payload["compliance_status"])
        status = payload.get("compliance_status")
        if status == ComplianceStatus.COMPLIANT:
            payload["overall_compliant"] = True
        elif status == ComplianceStatus.NON_COMPLIANT:
            payload["overall_compliant"] = False
        else:
            payload["overall_compliant"] = None
        return payload


class GroundingVerificationOutput(BaseModel, frozen=True):
    verified: bool
    unverified_claims: list[str]
    notes: str
    citations: list[Citation]
    status: GroundingStatus = GroundingStatus.NOT_ASSESSABLE
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def infer_status(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        verified = payload.get("verified", False)
        status = payload.get("status")
        if status is None:
            payload["status"] = GroundingStatus.VERIFIED if verified else GroundingStatus.UNVERIFIED
            status = payload["status"]
        elif isinstance(status, str):
            status = GroundingStatus(status)
            payload["status"] = status
        if status != GroundingStatus.VERIFIED:
            payload["verified"] = False
        return payload

    @model_validator(mode="after")
    def _verified_requires_citation(self) -> "GroundingVerificationOutput":
        # A negative verification is specifically the case where evidence is
        # absent; it must be representable without fabricating a citation.
        # Positive verification still requires evidence.
        if self.verified and not self.citations:
            msg = "a grounding verification must carry at least one citation"
            raise ValueError(msg)
        return self


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
    compliance_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSABLE
    grounding_status: GroundingStatus = GroundingStatus.NOT_ASSESSABLE
    evidence_policy: str = "mechanically_verified"
    coverage: dict[str, int] = Field(default_factory=dict)
    # Review provenance is part of the report contract, not hidden in graph
    # state. These values are copied from the server-authorized review fact.
    self_reviewed: bool = False
    reviewer_id: str | None = None
    reviewer_role: str | None = None
    contract_indices: list[int] = Field(default_factory=list)

    @field_validator("citations")
    @classmethod
    def _citations_non_empty(cls, value: list[Citation]) -> list[Citation]:
        # D-9: an uncited report reads as grounded. An empty citation list must
        # fail validation, not flow downstream as a legitimate answer.
        if not value:
            msg = "a final report must carry at least one citation"
            raise ValueError(msg)
        return value


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
# State schema versioning — one constant governs writing and reading (D-5).
# Bump STATE_SCHEMA_VERSION in the same commit that changes LegalAgentState;
# extend hydrate_state so older recognised shapes upgrade deterministically.
# ---------------------------------------------------------------------------

STATE_SCHEMA_VERSION: Final[int] = 7

# States persisted before the schema_version field existed carry no key at
# all; version 1 is also accepted as the recognised legacy shape and
# backfilled with fields introduced by the planner replay and analysis-safety
# fixes. These changes share one persisted-state migration boundary so the
# service's legacy-resume path can safely upgrade the old planner checkpoint.
# Existing values always win — defaults only fill what is missing.
_LEGACY_STATE_DEFAULTS: dict[str, Any] = {
    "user_id": "",
    "thread_id": "",
    "correlation_id": "",
    "run_id": "",
    "checkpoint_id": None,
    "doc_id": "",
    "primary_doc_id": None,
    "allowed_doc_ids": [],
    "tenant_id": "",
    "legal_roles": [],
    "allowed_operations": [],
    "capabilities": [],
    "document_versions": [],
    # Legacy checkpoints have no trustworthy tenant policy; fail closed.
    "self_review_allowed": False,
    "self_reviewed": False,
    "document_text": None,
    "canonical_segments": [],
    "canonical_structure_ready": False,
    "legacy_structure_fallback": False,
    "ingestion_state": [],
    "messages": [],
    "user_query": "",
    "qna_confidence": 0.0,
    "plan": [],
    "current_step": 0,
    "plan_approved": False,
    "plan_rejection_feedback": None,
    "orchestrator_action": None,
    "normalized_document": None,
    "segments": [],
    "extracted_entities": [],
    "relationships": [],
    "risk_analysis": None,
    "compliance_result": None,
    "grounding": None,
    "evidence_registry": {},
    "claim_ledger": [],
    "coverage": {},
    "evidence_policy": "mechanically_verified",
    "human_review": None,
    "final_report": None,
    "long_term_refs": [],
    "working_memory": {},
    "deep_research_results": None,
    "status": WorkflowStatus.INITIALIZED,
    "errors": [],
    "retry_count": 0,
    "permissions": {},
    "budget_ledger": BudgetLedger(),
}


class StateSchemaVersionError(ValueError):
    """Persisted agent state carries a schema version this runtime refuses."""


def hydrate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Version-check persisted state before any reasoning step reads it.

    Matching version passes through unchanged; the recognised legacy shape
    (version 0 / 1 / missing key) is upgraded deterministically; anything else
    is refused with a typed error naming both versions. Pure — never mutates
    the input dict.
    """
    recorded = state.get("schema_version")
    if recorded == STATE_SCHEMA_VERSION and not isinstance(state.get("budget_ledger"), dict):
        return state
    if recorded is not None and recorded not in {0, 1, 2, 3, 4, 5, 6}:
        msg = (
            f"refusing agent state: recorded schema_version={recorded!r} "
            f"cannot be loaded by this runtime "
            f"(expected {STATE_SCHEMA_VERSION!r})"
        )
        raise StateSchemaVersionError(msg)
    migrated = {**_LEGACY_STATE_DEFAULTS, **state}
    # v0-v2 only carried one document. Preserve that checkpoint's exact scope
    # rather than treating an absent allowlist as unrestricted retrieval.
    legacy_doc_id = migrated.get("doc_id")
    allowed = migrated.get("allowed_doc_ids")
    if not isinstance(allowed, list) or not allowed:
        migrated["allowed_doc_ids"] = [legacy_doc_id] if legacy_doc_id else []
    if not migrated.get("primary_doc_id"):
        migrated["primary_doc_id"] = legacy_doc_id or (
            migrated["allowed_doc_ids"][0] if migrated["allowed_doc_ids"] else None
        )
    raw_ingestion = migrated.get("ingestion_state")
    if not isinstance(raw_ingestion, list) or not raw_ingestion:
        migrated["ingestion_state"] = initial_ingestion_state(migrated["allowed_doc_ids"])
    raw_budget = migrated.get("budget_ledger")
    if isinstance(raw_budget, dict):
        try:
            migrated["budget_ledger"] = BudgetLedger.model_validate(raw_budget)
        except ValueError:
            # An unreadable ledger is not a reason to grant fresh budget.
            migrated["budget_ledger"] = BudgetLedger(exhausted=True)
    return {**migrated, "schema_version": STATE_SCHEMA_VERSION}


def legacy_resume_update(state: dict[str, Any]) -> dict[str, Any]:
    """Build the v2 update sent before directly resuming a v1 interrupt."""
    hydrated = hydrate_state(dict(state))
    declared = get_type_hints(LegalAgentState, include_extras=True)
    reducer_channels = {
        key
        for key, annotation in declared.items()
        if get_origin(annotation) is Annotated
        and any(
            metadata is operator.add or getattr(metadata, "__name__", None) == "add_messages"
            for metadata in get_args(annotation)[1:]
        )
    }
    return {
        key: value
        for key, value in hydrated.items()
        if key in declared and key not in reducer_channels
    }


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
    run_id: str
    checkpoint_id: str | None
    schema_version: int  # bump when TypedDict evolves; guards hydration node

    # --- Document reference ---
    doc_id: str
    primary_doc_id: str | None
    allowed_doc_ids: list[str]
    tenant_id: str
    legal_roles: list[str]
    allowed_operations: list[str]
    capabilities: list[str]
    document_versions: list[tuple[str, int, str | None]]
    self_review_allowed: bool
    self_reviewed: bool
    document_text: str | None  # populated by ingestion node
    canonical_segments: list[ClauseSegment]
    canonical_structure_ready: bool
    legacy_structure_fallback: bool
    ingestion_state: list[IngestionDocumentState]

    # --- Conversation messages (LangGraph message reducer) ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- Intent ---
    user_query: str
    qna_confidence: float  # 0.0-1.0; < 0.7 triggers HITL clarification

    # --- Planning ---
    plan: list[PlanStep]
    current_step: int
    plan_approved: bool
    plan_rejection_feedback: str | None
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
    evidence_registry: dict[str, EvidenceRef]
    claim_ledger: list[ClaimLedgerEntry]
    coverage: dict[str, int]
    evidence_policy: str

    # --- Human review ---
    human_review: HumanReviewOutput | None

    # --- Final ---
    final_report: FinalReport | None

    # --- Memory ---
    long_term_refs: list[str]  # cognee store namespace keys
    working_memory: dict[str, Any]  # ephemeral cross-node scratch space

    # --- Deep research ---
    deep_research_results: str | None  # batch results from deep_research node

    # --- Control ---
    status: WorkflowStatus
    errors: Annotated[list[AgentError], operator.add]
    retry_count: int
    permissions: dict[str, bool]
    # Persisted per-run usage; never reset on a HITL resume.
    budget_ledger: BudgetLedger


# Boundary schemas — state-compat input, Pydantic output at the graph edge.
# Interior state stays TypedDict per LangGraph best practice.
#
# NOTE: there is deliberately NO narrow Pydantic input schema. LangGraph
# VALIDATES a Pydantic input schema and then applies only the keys it
# declares — anything else is silently dropped before the first node
# (verified by probe: a `messages` outside the schema arrived as [] and a
# `status` outside it was absent at the first router). A state-compat
# TypedDict input applies channel updates instead, which is what the
# per-turn delta contract (§10) needs; request-shape validation belongs to
# the WS/HTTP DTO layer.
# ---------------------------------------------------------------------------


class LegalAgentOutputState(BaseModel, frozen=True):
    """Fields surfaced to callers after graph completion."""

    final_report: FinalReport | None = None
    status: WorkflowStatus
    errors: list[AgentError] = []


# Exported node names — used by service to filter astream_events
GRAPH_NODE_NAMES: frozenset[str] = frozenset(
    {
        "gateway",
        "qna",
        "orchestrator",
        "planner_generate",
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
        "deep_research",
    }
)
