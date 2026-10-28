"""Validated schema and loader for versioned retrieval golden sets.

The schema is a single evolving contract: ``legal_retrieval_v1`` rows remain
byte-compatible, while ``legal_retrieval_v2`` rows opt in to the extended
measurement vocabulary (case identity, spans, claims, adversarial markers,
ingestion states) through additive optional fields.  Cross-field rules V1 and
V3-V12 are enforced here (the no-evidence family rules that constituted
design-V2 are enforced via the shared ``NO_EVIDENCE_CASE_TYPES`` constant plus
the existing no-evidence validations, so no separate V2 rule id exists) so
that CI fails closed on incomplete v2 cases.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Literal

import aiofiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from returns.result import Failure, Result, Success

from app.observability.logging import logger
from app.utils.exceptions import InfrastructureException, NotFoundException

from .exceptions import MalformedGoldenRowError

type GoldenDocumentKind = Literal["contracts", "statutes", "judgments", "filings"]
type GoldenCaseType = Literal[
    "direct_answer",
    "multi_document",
    "multi_hop",
    "unanswerable",
    "statute_not_in_corpus",
    "conflicting_clauses",
    "near_miss_clause",
    "definition_exception_trap",
    "injection_content",
    "cross_tenant_isolation",
    "stale_version",
    "amendment_overrides",
    "no_evidence_vs_ingestion_failure",
    "index_not_ready",
    "multi_contract_boundary",
    "ocr_semantic_flip",
    "table_misalignment",
    "footnote_as_operative",
    "annexure_as_main",
    "wrong_jurisdiction",
    "wrong_date_version_authority",
    "partial_coverage_answer",
    "handle_integrity",
]
type RetrievalRelevance = Literal["relevant", "partially_relevant", "irrelevant", "not_applicable"]
type ClaimEntailment = Literal[
    "entailed", "partial", "contradicted", "unsupported", "not_applicable"
]
type ProvenanceSourceType = Literal["public", "synthetic", "client"]
type PermissionStatus = Literal["public_terms_verified", "synthetic_owned", "written_permission"]
type TermsStatus = Literal["verified", "not_required", "pending"]
type HumanReviewStatus = Literal["pending", "engineer_reviewed", "sme_reviewed", "adjudicated"]
type MeasurementFamily = Literal[
    "retrieval", "grounding", "abstention", "scope", "ingestion", "contract_identity", "report"
]
type GoldenGrader = Literal["deterministic", "sme", "hybrid"]
type GoldenOcrProfile = Literal["clean", "light_noise", "heavy_noise"]
type PairRole = Literal["control", "treatment"]
type ExpectedSupport = Literal["entailed", "partial", "contradicted", "unsupported"]
type ClaimGradability = Literal["deterministic", "sme"]
type AdversarialMarkerKind = Literal[
    "prompt_injection", "fake_system_message", "fake_tool_output", "urgent_directive"
]
# Mirrors the EvidenceStatus vocabulary of the documents feature; redeclared
# locally so the evaluation schema never imports the feature layer.
type GoldenEvidenceStatus = Literal[
    "ok", "no_evidence", "index_not_ready", "ingestion_failed", "unauthorized", "stale_version"
]
type GoldenEvidencePolicy = Literal["mechanically_verified", "graph_indexed_only"]
type GoldenIngestionState = Literal[
    "ready", "completed_with_warnings", "index_not_ready", "failed", "unauthorized"
]
type QualityTier = Literal["candidate", "release"]
type AnnotationRole = Literal["engineer", "legal_sme"]
type OcrQualityImpact = Literal["none", "meaning_preserved", "meaning_at_risk", "meaning_changed"]
type AdjudicationMethod = Literal["independent_sme"]
type AdjudicationStatus = Literal[
    "not_required", "pending_sme", "adjudicated", "waived_with_reason"
]

GOLDEN_PROVENANCE_SCHEMA_VERSION = "golden_provenance_v1"
GOLDEN_SET_V2_VERSION = "legal_retrieval_v2"

# Case types whose correct behaviour is to abstain because no usable evidence
# can exist for the query.  The two ingestion-family categories extend the
# original v1 pair without changing any v1 rule.
NO_EVIDENCE_CASE_TYPES: frozenset[str] = frozenset(
    {
        "unanswerable",
        "statute_not_in_corpus",
        "no_evidence_vs_ingestion_failure",
        "index_not_ready",
    }
)
# Case types where per-policy expected outcomes are meaningful.
POLICY_OVERRIDE_CASE_TYPES: frozenset[str] = frozenset(
    {"no_evidence_vs_ingestion_failure", "index_not_ready"}
)
# Case types that must declare exactly one coverage expectation.
COVERAGE_CASE_TYPES: frozenset[str] = frozenset(
    {"no_evidence_vs_ingestion_failure", "index_not_ready", "partial_coverage_answer"}
)
# Case types that measure document-version authority.
VERSION_AUTHORITY_CASE_TYPES: frozenset[str] = frozenset(
    {"stale_version", "amendment_overrides", "wrong_date_version_authority"}
)
_ABSTENTION_EVIDENCE_STATUSES: frozenset[str] = frozenset(
    {"no_evidence", "index_not_ready", "ingestion_failed", "unauthorized", "stale_version"}
)
_COMPLETABLE_ADJUDICATION_STATUSES: frozenset[str] = frozenset(
    {"adjudicated", "not_required", "waived_with_reason"}
)
# Parse-time aliases keep v1 case_type values valid while the canonical
# vocabulary above becomes the stored form.
_CASE_TYPE_ALIASES: dict[str, GoldenCaseType] = {
    "standard": "direct_answer",
    "near_miss": "near_miss_clause",
}
_MODEL_ADJUDICATOR_PATTERN = re.compile(
    r"(?<![A-Za-z])(gpt|claude|gemini|llm|model)(?![A-Za-z])", re.IGNORECASE
)
# Compound model names embed guarded tokens inside letters (ChatGPT, AutoGPT);
# no human name contains these, so they are matched without boundaries.
_MODEL_ADJUDICATOR_COMPOUND_PATTERN = re.compile(
    r"chatgpt|autogpt|openai|anthropic|deepseek|copilot", re.IGNORECASE
)


def _adjudicator_is_model(value: str) -> bool:
    """True when a value names a model under either the guarded or compound rule."""
    return bool(
        _MODEL_ADJUDICATOR_PATTERN.search(value)
        or _MODEL_ADJUDICATOR_COMPOUND_PATTERN.search(value)
    )
_REDISTRIBUTION_DENY_TOKENS = (
    "disallow",
    "forbid",
    "not_permitted",
    "no_redistribution",
    "prohibit",
    "denied",
)

_ADJUDICATOR_REQUIRED = "A completed disagreement requires an adjudicator"
_WAIVER_RATIONALE_MIN_LENGTH = 20
_WAIVER_REQUIRES_RATIONALE = (
    "Waived-with-reason requires a rationale of at least 20 characters"
)
_WAIVER_REQUIRES_ADJUDICATOR = "Waived-with-reason requires a named human adjudicator"
_SOURCE_FLAGS_MISMATCH = "Public/synthetic indicators do not match source_type"
_PERMISSION_MISMATCH = "permission_status is incompatible with source_type"
_TERMS_REFERENCE_REQUIRED = "Verified license/terms status requires a reference"
_CLIENT_TERMS_REQUIRED = "Client material requires verified license/terms status"
_PUBLIC_TERMS_PENDING = "Public material with unreviewed terms is not admissible"
_ANSWERABLE_CHUNKS_REQUIRED = "Answerable and near-miss cases need expected chunk identifiers"
_NO_EVIDENCE_RELEVANCE = "No-evidence cases must use not_applicable retrieval relevance"
_NO_EVIDENCE_ENTAILMENT = "No-evidence cases must use not_applicable claim entailment"
_CASE_ID_PREFIX_MISMATCH = "case_id must start with the canonical case_type slug"
_ABSTENTION_CONTRADICTION = (
    "Abstention cases cannot declare expected claims and need a non-ok expected_evidence_status"
)
_ADVERSARIAL_INCOMPLETE = (
    "Adversarial cases need at least one adversarial marker and a must-not-include answer list"
)
_CONTRACT_INDEX_REQUIRED = "Multi-contract boundary cases need expected_contract_indexes"
_VERSION_AUTHORITY_INCOMPLETE = (
    "Version-authority cases need expected_document_versions and stale or forbidden evidence"
)
_COVERAGE_EXPECTATION_REQUIRED = (
    "Coverage-family cases must declare exactly one of expected_coverage "
    "or expected_coverage_ratio_min"
)
_POLICY_OVERRIDE_MISPLACED = (
    "policy_overrides are only valid on no_evidence_vs_ingestion_failure and index_not_ready rows"
)
_FORBIDDEN_EXPECTED_OVERLAP = "Forbidden expectations must be disjoint from expected ones"
_SME_CLAIM_WITH_DETERMINISTIC_GRADER = "Deterministic graders cannot grade sme-gradability claims"
_PARTIAL_PAIR_FIELDS = "Paired rows must declare pair_id, pair_variable and pair_role together"
_MODEL_ADJUDICATOR_FORBIDDEN = "Adjudicator must name a human reviewer, not a model"
_ADJUDICATED_REQUIRES_HUMAN = "Adjudicated records require a named human adjudicator"


class ExpectedSpan(BaseModel):
    """A verbatim quote that a correct answer must rest on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    contract_index: int | None = None
    page_no: int = Field(ge=1)
    quote: str = Field(min_length=1)


class ExpectedClaim(BaseModel):
    """One atomic claim with its expected support and grading contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    expected_support: ExpectedSupport
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    gradability: ClaimGradability


class AdversarialMarker(BaseModel):
    """Location of an injected or misleading artefact inside a document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    page_no: int = Field(ge=1)
    kind: AdversarialMarkerKind


class ExpectedCoverage(BaseModel):
    """Ingestion coverage counts expected for a coverage-family case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    documents_requested: int = Field(ge=0)
    documents_ready: int = Field(ge=0)
    documents_with_warnings: int = Field(ge=0)
    documents_failed: int = Field(ge=0)
    documents_not_ready: int = Field(ge=0)


class ExpectedIngestionState(BaseModel):
    """Per-document ingestion state a case is measured against."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    state: GoldenIngestionState


class PolicyOverride(BaseModel):
    """Expected outcome under one alternative evidence policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_abstention: bool
    expected_evidence_status: GoldenEvidenceStatus


class DoubleLabel(BaseModel):
    """Optional second-pass labels and adjudication record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_relevance: RetrievalRelevance
    claim_entailment: ClaimEntailment
    agreement: Literal["agree", "disagree"]
    adjudication_status: Literal["not_required", "pending", "complete"]
    adjudicator: str | None = None

    @model_validator(mode="after")
    def require_adjudicator_for_complete_disagreement(self) -> DoubleLabel:
        if (
            self.agreement == "disagree"
            and self.adjudication_status == "complete"
            and not self.adjudicator
        ):
            raise ValueError(_ADJUDICATOR_REQUIRED)
        return self


class AnnotationRecord(BaseModel):
    """One independent annotation pass over a golden item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    annotator_id: str = Field(min_length=1)
    role: AnnotationRole
    timestamp_utc: str = Field(min_length=1)
    labels: dict[str, object] = Field(default_factory=dict)
    supporting_spans: list[ExpectedSpan] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    ambiguity_notes: str = ""
    ocr_quality_impact: OcrQualityImpact
    harness_version: str = Field(min_length=1)


class AdjudicationRecord(BaseModel):
    """Human adjudication outcome for a disputed or release-bound item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AdjudicationStatus
    adjudicator: str | None = None
    method: AdjudicationMethod | None = None
    decided_at: str | None = None
    rationale: str | None = None
    checklist: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_human_adjudicator(self) -> AdjudicationRecord:
        if self.adjudicator and _adjudicator_is_model(self.adjudicator):
            raise ValueError(_MODEL_ADJUDICATOR_FORBIDDEN)
        if self.status == "adjudicated" and not self.adjudicator:
            raise ValueError(_ADJUDICATED_REQUIRES_HUMAN)
        if self.status == "waived_with_reason":
            # A waiver satisfies the release gate, so it must be as accountable
            # as an adjudication: a substantive rationale and a named human.
            # The model-name guard above already covers the adjudicator value.
            if self.rationale is None or len(self.rationale.strip()) < _WAIVER_RATIONALE_MIN_LENGTH:
                raise ValueError(_WAIVER_REQUIRES_RATIONALE)
            if not self.adjudicator:
                raise ValueError(_WAIVER_REQUIRES_ADJUDICATOR)
        return self


class AnnotatorRoles(BaseModel):
    """Role attribution for the two independent evaluation dimensions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engineer_relevance: str = Field(min_length=1)
    legal_sme_entailment: str = Field(min_length=1)


class GoldenItemProvenance(BaseModel):
    """Source and permission record required before a gold item is CI-valid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: ProvenanceSourceType
    source_reference: str = Field(min_length=1)
    permission_status: PermissionStatus
    license_terms_status: TermsStatus
    license_terms_reference: str | None = None
    is_public: bool
    is_synthetic: bool
    human_review_status: HumanReviewStatus
    annotators: AnnotatorRoles
    double_label: DoubleLabel | None = None
    quality_tier: QualityTier = "candidate"
    annotations: list[AnnotationRecord] = Field(default_factory=list)
    adjudication: AdjudicationRecord | None = None

    @model_validator(mode="after")
    def validate_source_and_permission(self) -> GoldenItemProvenance:
        expected_flags = {
            "public": (True, False, "public_terms_verified"),
            "synthetic": (False, True, "synthetic_owned"),
            "client": (False, False, "written_permission"),
        }
        expected_public, expected_synthetic, expected_permission = expected_flags[self.source_type]
        if (self.is_public, self.is_synthetic) != (expected_public, expected_synthetic):
            raise ValueError(_SOURCE_FLAGS_MISMATCH)
        if self.permission_status != expected_permission:
            raise ValueError(_PERMISSION_MISMATCH)
        if self.license_terms_status == "verified" and not self.license_terms_reference:
            raise ValueError(_TERMS_REFERENCE_REQUIRED)
        if self.source_type == "client" and self.license_terms_status != "verified":
            raise ValueError(_CLIENT_TERMS_REQUIRED)
        if self.source_type == "public" and self.license_terms_status == "pending":
            raise ValueError(_PUBLIC_TERMS_PENDING)
        return self


class GoldenQuery(BaseModel):
    """One human-curated retrieval expectation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    document_kind: GoldenDocumentKind
    difficulty: Literal["easy", "medium", "hard"]
    notes: str
    awaiting_sme_expansion: bool
    case_type: GoldenCaseType = "direct_answer"
    retrieval_relevance: RetrievalRelevance = "relevant"
    claim_entailment: ClaimEntailment = "entailed"
    provenance: GoldenItemProvenance | None = None
    # v2 measurement vocabulary: additive, defaulted, v1 rows never set them.
    case_id: str | None = None
    measurement_family: MeasurementFamily | None = None
    grader: GoldenGrader | None = None
    ocr_profile: GoldenOcrProfile = "clean"
    pair_id: str | None = None
    pair_variable: str | None = None
    pair_role: PairRole | None = None
    expected_spans: list[ExpectedSpan] = Field(default_factory=list)
    expected_contract_indexes: list[int] = Field(default_factory=list)
    expected_document_versions: list[str] = Field(default_factory=list)
    stale_document_ids: list[str] = Field(default_factory=list)
    forbidden_document_ids: list[str] = Field(default_factory=list)
    forbidden_chunk_ids: list[str] = Field(default_factory=list)
    forbidden_contract_indexes: list[int] = Field(default_factory=list)
    allowed_doc_ids: list[str] = Field(default_factory=list)
    expected_claims: list[ExpectedClaim] = Field(default_factory=list)
    expected_answer_must_include: list[str] = Field(default_factory=list)
    expected_answer_must_not_include: list[str] = Field(default_factory=list)
    adversarial: bool = False
    adversarial_markers: list[AdversarialMarker] = Field(default_factory=list)
    expected_abstention: bool = False
    expected_evidence_status: GoldenEvidenceStatus | None = None
    expected_coverage: ExpectedCoverage | None = None
    expected_coverage_ratio_min: float | None = Field(default=None, ge=0, le=1)
    expected_evidence_policy: GoldenEvidencePolicy = "mechanically_verified"
    expected_ingestion_states: list[ExpectedIngestionState] = Field(default_factory=list)
    policy_overrides: dict[str, PolicyOverride] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_case_type_aliases(cls, data: object) -> object:
        """Map v1 case_type values onto the canonical v2 vocabulary."""
        if isinstance(data, dict):
            case_type = data.get("case_type")
            if isinstance(case_type, str) and case_type in _CASE_TYPE_ALIASES:
                return {**data, "case_type": _CASE_TYPE_ALIASES[case_type]}
        return data

    @model_validator(mode="after")
    def validate_case_expectations(self) -> GoldenQuery:
        no_evidence_case = self.case_type in NO_EVIDENCE_CASE_TYPES
        if no_evidence_case and (self.expected_chunk_ids or self.expected_document_ids):
            message = f"{self.case_type} cases cannot declare expected evidence"
            raise ValueError(message)
        if not no_evidence_case and not self.expected_chunk_ids:
            raise ValueError(_ANSWERABLE_CHUNKS_REQUIRED)
        if no_evidence_case and self.retrieval_relevance != "not_applicable":
            raise ValueError(_NO_EVIDENCE_RELEVANCE)
        if no_evidence_case and self.claim_entailment != "not_applicable":
            raise ValueError(_NO_EVIDENCE_ENTAILMENT)
        self._validate_measurement_contract()
        return self

    def _validate_measurement_contract(self) -> None:
        """Enforce cross-field rules V1 and V3-V12 (V1 uniqueness is set-level).

        There is no V2 rule id: the no-evidence family rules that constituted
        design-V2 run through ``NO_EVIDENCE_CASE_TYPES`` in
        ``validate_case_expectations`` instead.
        """
        # V1: slug prefix of case_id matches the canonical case_type.
        if self.case_id is not None and not self.case_id.startswith(self.case_type):
            raise ValueError(_CASE_ID_PREFIX_MISMATCH)
        # V3: abstention rows forbid claims and need a non-ok evidence status.
        if self.expected_abstention:
            abstention_status = self.expected_evidence_status
            if self.expected_claims or (
                abstention_status is None or abstention_status not in _ABSTENTION_EVIDENCE_STATUSES
            ):
                raise ValueError(_ABSTENTION_CONTRADICTION)
        # V5: adversarial rows need markers and a must-not-include list.
        if self.adversarial and (
            not self.adversarial_markers or not self.expected_answer_must_not_include
        ):
            raise ValueError(_ADVERSARIAL_INCOMPLETE)
        # V6: multi-contract boundary cases name the expected contract indexes.
        if self.case_type == "multi_contract_boundary" and not self.expected_contract_indexes:
            raise ValueError(_CONTRACT_INDEX_REQUIRED)
        # V7: version-authority cases need versions plus stale/forbidden evidence.
        if self.case_type in VERSION_AUTHORITY_CASE_TYPES and (
            not self.expected_document_versions
            or not (self.stale_document_ids or self.forbidden_chunk_ids)
        ):
            raise ValueError(_VERSION_AUTHORITY_INCOMPLETE)
        # V8: coverage-family cases declare exactly one coverage expectation.
        if self.case_type in COVERAGE_CASE_TYPES and (
            (self.expected_coverage is None) == (self.expected_coverage_ratio_min is None)
        ):
            raise ValueError(_COVERAGE_EXPECTATION_REQUIRED)
        # Policy overrides only make sense for the ingestion-failure family.
        if self.policy_overrides and self.case_type not in POLICY_OVERRIDE_CASE_TYPES:
            raise ValueError(_POLICY_OVERRIDE_MISPLACED)
        self._validate_expectation_sets()

    def _validate_expectation_sets(self) -> None:
        """Enforce the set-shape rules V9, V11, and the row half of V12."""
        # V9: forbidden expectations must be disjoint from expected ones.
        if set(self.forbidden_document_ids) & set(self.expected_document_ids):
            raise ValueError(_FORBIDDEN_EXPECTED_OVERLAP)
        if set(self.forbidden_chunk_ids) & set(self.expected_chunk_ids):
            raise ValueError(_FORBIDDEN_EXPECTED_OVERLAP)
        if set(self.forbidden_contract_indexes) & set(self.expected_contract_indexes):
            raise ValueError(_FORBIDDEN_EXPECTED_OVERLAP)
        # V11: deterministic graders cannot grade sme-gradability claims.
        if self.grader == "deterministic" and any(
            claim.gradability == "sme" for claim in self.expected_claims
        ):
            raise ValueError(_SME_CLAIM_WITH_DETERMINISTIC_GRADER)
        # V12 (row half): paired rows declare all three pair fields.
        pair_fields = (self.pair_id, self.pair_variable, self.pair_role)
        if any(field is not None for field in pair_fields) and not all(
            field is not None for field in pair_fields
        ):
            raise ValueError(_PARTIAL_PAIR_FIELDS)


class GoldenSet(BaseModel):
    """A comparable, explicitly versioned collection of golden queries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    queries: list[GoldenQuery] = Field(min_length=1)
    provenance_schema_version: str | None = None


class ArtifactProvenance(BaseModel):
    """Provenance record for one acquired corpus artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    title: str | None = None
    issuer: str | None = None
    source_url: str | None = None
    terms_licence_url: str | None = None
    jurisdiction: str | None = None
    document_date: str | None = None
    access_date: str | None = None
    source_type: str | None = None
    original_filename: str | None = None
    mime_type: str | None = None
    sha256: str | None = None
    file_size_bytes: int | None = Field(default=None, ge=0)
    classification: str | None = None
    permission_status: str | None = None
    redistribution_status: str | None = None
    model_processing_status: str | None = None
    ocr_status: str | None = None
    ocr_engine: str | None = None
    characteristics: dict[str, object] = Field(default_factory=dict)
    document_version_identity: str | None = None
    contract_boundaries: list[dict[str, object]] = Field(default_factory=list)
    page_map_file: str | None = None
    attribution_string: str | None = None
    unresolved_rights_notes: str | None = None


class SmeRequirements(BaseModel):
    """Declared human sign-off requirements for the golden set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    independence: str = Field(min_length=1)
    commitment: str = Field(min_length=1)
    named_adjudicator: str | None = None
    adjudication_status: str = Field(default="pending_sme", min_length=1)

    @model_validator(mode="after")
    def named_adjudicator_must_be_human(self) -> SmeRequirements:
        """Reject model-like names with the same rule as AdjudicationRecord."""
        if self.named_adjudicator and _adjudicator_is_model(self.named_adjudicator):
            raise ValueError(_MODEL_ADJUDICATOR_FORBIDDEN)
        return self


class GoldenProvenanceManifest(BaseModel):
    """Manifest binding a golden set to its source artifacts and policies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = Field(min_length=1)
    golden_set_version: str = Field(min_length=1)
    artifact: str = Field(min_length=1)
    source_policy: dict[str, object] = Field(default_factory=dict)
    annotation_policy: dict[str, object] = Field(default_factory=dict)
    sme_requirements: SmeRequirements | None = None
    per_artifact: list[ArtifactProvenance] = Field(default_factory=list)
    corpus_manifest_path: str | None = None
    notes: str | None = None


class GoldenSetProvenanceError(ValueError):
    """Raised when a set contains unknown, unreviewed, or inadmissible provenance."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


type GoldenSetLoadResult = Result[GoldenSet, MalformedGoldenRowError]


class GoldenSetNotFoundException(NotFoundException):
    """The requested golden-set artifact does not exist."""

    def __init__(self, path: Path) -> None:
        super().__init__(resource="Golden set", identifier=str(path))


class GoldenSetReadException(InfrastructureException):
    """The requested golden-set artifact exists but cannot be read."""

    def __init__(self, path: Path, cause: OSError | UnicodeDecodeError) -> None:
        super().__init__(
            detail=f"Unable to read golden set: {path}",
            retryable=False,
            original_exc=cause,
        )


def _malformed(
    *, row_index: int, message: str, details: dict[str, object] | None = None
) -> Failure[MalformedGoldenRowError]:
    return Failure(
        MalformedGoldenRowError(
            message=message,
            row_index=row_index,
            details={"row_index": row_index, **(details or {})},
            source="golden_set_loader",
        )
    )


def manifest_path_for(golden_set_path: Path) -> Path:
    """Return the provenance manifest path paired with a golden-set JSONL."""
    return golden_set_path.with_name(f"{golden_set_path.stem}.manifest.json")


def referenced_document_ids(query: GoldenQuery) -> set[str]:
    """Collect every document identifier a case references by id.

    Documents named only in ``expected_ingestion_states`` count too: an
    ingestion-failure case measures a document it never expects to retrieve,
    so omitting it here would let a case reference an undeclared document.
    """
    referenced: set[str] = set(query.expected_document_ids)
    referenced.update(query.allowed_doc_ids)
    referenced.update(query.stale_document_ids)
    referenced.update(query.forbidden_document_ids)
    referenced.update(span.document_id for span in query.expected_spans)
    referenced.update(marker.document_id for marker in query.adversarial_markers)
    referenced.update(state.document_id for state in query.expected_ingestion_states)
    return referenced


def validate_golden_set(
    golden_set: GoldenSet,
    *,
    manifest: GoldenProvenanceManifest | None = None,
) -> None:
    """Enforce provenance policy for CI and evaluation execution.

    ``load_golden_set`` remains a schema loader for backwards-compatible
    programmatic callers. This explicit gate prevents legacy rows without a
    provenance record from being silently treated as approved evaluation data.
    ``manifest`` enables the manifest-dependent conditions (named SME
    adjudication for release-tier rows) and is optional for compatibility.
    """
    issues: list[str] = []
    if golden_set.provenance_schema_version != GOLDEN_PROVENANCE_SCHEMA_VERSION:
        issues.append(
            f"golden set is missing provenance_schema_version={GOLDEN_PROVENANCE_SCHEMA_VERSION}"
        )
    enforce_v2_rules = golden_set.version == GOLDEN_SET_V2_VERSION
    case_ids: dict[str, int] = {}
    pair_groups: dict[str, list[GoldenQuery]] = {}
    for row_index, query in enumerate(golden_set.queries, start=2):
        provenance = query.provenance
        if provenance is None:
            issues.append(f"row {row_index}: provenance is missing and needs review")
        else:
            _validate_row_provenance(row_index, query, provenance, manifest, issues)
        if query.case_id is not None:
            first_row = case_ids.setdefault(query.case_id, row_index)
            if first_row != row_index:
                issues.append(
                    f"row {row_index}: case_id {query.case_id!r} duplicates row {first_row}"
                )
        if query.pair_id is not None:
            pair_groups.setdefault(query.pair_id, []).append(query)
        if enforce_v2_rules:
            _validate_v2_case(row_index, query, issues)
    for pair_id, group in pair_groups.items():
        roles = {query.pair_role for query in group}
        variables = {query.pair_variable for query in group}
        if len(group) != 2 or roles != {"control", "treatment"} or len(variables) != 1:
            issues.append(
                f"pair {pair_id}: paired rows must come in exactly two with opposite "
                "pair_role values and one shared pair_variable"
            )
    if issues:
        raise GoldenSetProvenanceError(issues)


def _validate_row_provenance(
    row_index: int,
    query: GoldenQuery,
    provenance: GoldenItemProvenance,
    manifest: GoldenProvenanceManifest | None,
    issues: list[str],
) -> None:
    """Row-level provenance conditions shared by every set version."""
    if provenance.permission_status not in {
        "public_terms_verified",
        "synthetic_owned",
        "written_permission",
    }:
        issues.append(f"row {row_index}: permission status is not admissible")
    if provenance.license_terms_status == "pending":
        issues.append(f"row {row_index}: license/terms review is pending")
    if provenance.source_type == "client" and provenance.permission_status != "written_permission":
        issues.append(f"row {row_index}: client material lacks written permission")
    if (
        provenance.human_review_status == "sme_reviewed"
        and provenance.annotators.legal_sme_entailment in {"", "pending"}
    ):
        issues.append(f"row {row_index}: SME-reviewed item has no legal SME annotator")
    if query.case_type in NO_EVIDENCE_CASE_TYPES and query.expected_chunk_ids:
        issues.append(f"row {row_index}: no-evidence case has expected chunks")
    # Condition 6: an SME approval claim needs a named human adjudication record.
    if provenance.human_review_status == "adjudicated" and (
        provenance.adjudication is None or provenance.adjudication.status != "adjudicated"
    ):
        issues.append(
            f"row {row_index}: adjudicated review status requires a matching adjudication record"
        )
    if (
        provenance.quality_tier == "release"
        and provenance.adjudication is not None
        and provenance.adjudication.status == "adjudicated"
    ):
        named_adjudicator = (
            manifest.sme_requirements.named_adjudicator
            if manifest is not None and manifest.sme_requirements is not None
            else None
        )
        # A blank or whitespace-only name is as good as no name at all.
        if not (named_adjudicator or "").strip():
            issues.append(
                f"row {row_index}: release-tier adjudication requires a named adjudicator "
                "in the manifest sme_requirements"
            )
    # Condition 9: release-quality cases must have completed human accountability.
    if provenance.quality_tier == "release":
        if (
            provenance.adjudication is None
            or provenance.adjudication.status not in _COMPLETABLE_ADJUDICATION_STATUSES
        ):
            issues.append(
                f"row {row_index}: release-quality case requires adjudication status of "
                "adjudicated, not_required, or waived_with_reason"
            )
        if provenance.human_review_status == "pending":
            issues.append(f"row {row_index}: release-quality case requires completed human review")


def _validate_v2_case(row_index: int, query: GoldenQuery, issues: list[str]) -> None:
    """Set-level v2 rules: V4 span/claim completeness and V10 ingestion coverage."""
    # V4: graded entailment requires concrete spans and claims.
    if query.claim_entailment != "not_applicable" and (
        not query.expected_spans or not query.expected_claims
    ):
        issues.append(
            f"row {row_index}: cases with claim entailment need expected_spans and expected_claims"
        )
    # V10: every referenced document must carry an expected ingestion state.
    ingested = {state.document_id for state in query.expected_ingestion_states}
    missing = sorted(referenced_document_ids(query) - ingested)
    if missing:
        issues.append(
            f"row {row_index}: documents missing from expected_ingestion_states: "
            f"{', '.join(missing)}"
        )


def _redistribution_disallows_storage(status: str | None) -> bool:
    """Whether a redistribution status forbids storing the raw artifact."""
    if not status:
        return False
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in _REDISTRIBUTION_DENY_TOKENS)


def _artifact_root(manifest: GoldenProvenanceManifest, manifest_dir: Path | None) -> Path:
    """Resolve the corpus root that artifact directories live under.

    A relative ``corpus_manifest_path`` is resolved against the manifest's own
    directory — recorded via ``manifest_dir`` by callers that know where the
    manifest was loaded from — so the audit is independent of the process
    working directory.  ``manifest_dir`` is ``None`` only for direct audit
    callers without a manifest path, which keep the historical cwd fallback.
    """
    base = manifest.corpus_manifest_path or "evals/corpus"
    root = Path(base)
    if root.is_absolute():
        return root
    return (manifest_dir if manifest_dir is not None else Path.cwd()) / root


def _artifact_file(root: Path, entry: ArtifactProvenance) -> Path | None:
    """Locate the raw original file for one artifact entry, if it exists."""
    directory = root / "artifacts" / entry.source_id
    if entry.original_filename:
        candidate = directory / entry.original_filename
        return candidate if candidate.is_file() else None
    for candidate in sorted(directory.glob("original.*")):
        if candidate.is_file():
            return candidate
    return None


async def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    async with aiofiles.open(path, "rb") as stream:
        while chunk := await stream.read(1024 * 512):
            digest.update(chunk)
    return digest.hexdigest()


async def audit_golden_manifest(
    manifest: GoldenProvenanceManifest,
    golden_set: GoldenSet,
    *,
    manifest_path: Path | None = None,
) -> list[str]:
    """Audit manifest artifacts against the files on disk and the gold rows.

    Returns CI-failing issues.  The golden-set-version consistency check always
    runs, with or without artifact entries.  Missing original files are skipped
    with a recorded warning; present files must match their declared sha256,
    and v2 sets may only reference documents declared in ``per_artifact``.  A
    v2 manifest without ``per_artifact`` entries is itself an issue: it would
    silently disable the provenance audit.  ``manifest_path`` anchors the
    resolution of a relative ``corpus_manifest_path`` to the manifest's own
    directory; callers that loaded the manifest from disk should pass it.
    """
    issues: list[str] = []
    if manifest.golden_set_version != golden_set.version:
        issues.append(
            f"manifest golden_set_version {manifest.golden_set_version!r} does not match "
            f"golden-set version {golden_set.version!r}"
        )
    if not manifest.per_artifact:
        if golden_set.version == GOLDEN_SET_V2_VERSION:
            issues.append("v2 manifest has empty per_artifact; provenance audit disabled")
        # v1 manifests carry no per_artifact entries; with nothing declared
        # there is nothing further to audit for them.
        return issues
    source_ids = {entry.source_id for entry in manifest.per_artifact}
    artifact_root = _artifact_root(
        manifest, manifest_path.parent if manifest_path is not None else None
    )
    for entry in manifest.per_artifact:
        issues.extend(await _audit_artifact_entry(entry, artifact_root))
    if golden_set.version == GOLDEN_SET_V2_VERSION:
        for row_index, query in enumerate(golden_set.queries, start=2):
            unknown = sorted(referenced_document_ids(query) - source_ids)
            if unknown:
                issues.append(
                    f"row {row_index}: document identifiers not present in manifest "
                    f"per_artifact: {', '.join(unknown)}"
                )
    return issues


async def _audit_artifact_entry(
    entry: ArtifactProvenance,
    artifact_root: Path,
) -> list[str]:
    """Audit one artifact entry: checksum, source URL, classification, storage."""
    label = f"artifact {entry.source_id}"
    issues: list[str] = []
    if not entry.sha256:
        issues.append(f"{label}: sha256 checksum is missing")
    if not entry.source_url:
        issues.append(f"{label}: source_url is missing")
    classification = (entry.classification or "").strip().lower()
    if classification == "confidential" or (
        classification == "client" and entry.permission_status != "written_permission"
    ):
        issues.append(
            f"{label}: classification {entry.classification!r} is not admissible "
            "without written permission"
        )
    disallowed = _redistribution_disallows_storage(entry.redistribution_status)
    artifact_file = _artifact_file(artifact_root, entry)
    if artifact_file is not None:
        if disallowed:
            issues.append(
                f"{label}: raw original {artifact_file.name} is stored although "
                f"redistribution status {entry.redistribution_status!r} disallows it"
            )
        elif entry.sha256:
            actual = await _sha256_of(artifact_file)
            if actual != entry.sha256:
                issues.append(
                    f"{label}: sha256 mismatch for {artifact_file.name} "
                    f"(expected {entry.sha256}, found {actual})"
                )
    elif not disallowed:
        sys.stderr.write(f"warning: {label}: original file not found; checksum skipped\n")
    return issues


async def _load_manifest(manifest_path: Path) -> GoldenProvenanceManifest:
    payload: str
    async with aiofiles.open(manifest_path, encoding="utf-8") as stream:
        payload = await stream.read()
    return GoldenProvenanceManifest.model_validate(json.loads(payload))


async def load_validated_golden_set(path: Path) -> GoldenSetLoadResult:
    """Load a golden set and fail if its provenance is not CI-admissible.

    When a sibling ``<stem>.manifest.json`` exists it is parsed and audited:
    artifact checksums are recomputed, permissions are re-checked, and (for
    ``legal_retrieval_v2``) row document references are cross-referenced
    against the manifest's per_artifact source ids.  A ``legal_retrieval_v2``
    set without its sibling manifest fails closed in every consumer of this
    loader, not only in the validate CLI: without the manifest the provenance
    audit cannot run, so the load is rejected.  v1 sets keep loading with or
    without a manifest.
    """
    result = await load_golden_set(path)
    if isinstance(result, Failure):
        return result
    golden_set = result.unwrap()
    manifest: GoldenProvenanceManifest | None = None
    manifest_path = manifest_path_for(path)
    if manifest_path.is_file():
        try:
            manifest = await _load_manifest(manifest_path)
        except (OSError, ValueError) as exc:
            # MalformedGoldenRowError is a frozen pydantic model, not a
            # BaseException, so e.add_note() is unavailable here; the cause
            # is retained both in the structured error details and in the
            # internal diagnostic log. The API-facing message stays unchanged.
            logger.bind(component="evaluation", manifest=str(manifest_path)).opt(
                exception=True
            ).warning("golden_set_manifest_load_failed", error=str(exc))
            return _malformed(
                row_index=1,
                message=f"Unable to load golden-set manifest: {exc}",
                details={
                    "manifest": str(manifest_path),
                    "manifest_cause": f"{type(exc).__name__}: {exc}",
                },
            )
    elif golden_set.version == GOLDEN_SET_V2_VERSION:
        # Fail closed before provenance validation: without the manifest the
        # release-tier conditions in validate_golden_set would also produce
        # misleading issues, so the missing sibling is the root cause to report.
        return _malformed(
            row_index=1,
            message=(
                f"legal_retrieval_v2 golden set requires a sibling "
                f"{manifest_path.name} provenance manifest (expected at {manifest_path})"
            ),
        )
    try:
        validate_golden_set(golden_set, manifest=manifest)
    except GoldenSetProvenanceError as exc:
        first_issue = exc.issues[0] if exc.issues else "Invalid golden-set provenance"
        return _malformed(row_index=1, message=first_issue)
    if manifest is not None:
        issues = await audit_golden_manifest(manifest, golden_set, manifest_path=manifest_path)
        if issues:
            return _malformed(row_index=1, message=issues[0])
    return result


async def load_golden_set(path: Path) -> GoldenSetLoadResult:
    """Load metadata plus query rows from a JSON Lines golden-set artifact."""
    try:
        async with aiofiles.open(path, encoding="utf-8") as stream:
            lines = await stream.readlines()
    except FileNotFoundError as exc:
        raise GoldenSetNotFoundException(path) from exc
    except OSError as exc:
        raise GoldenSetReadException(path, exc) from exc
    except UnicodeDecodeError as exc:
        # UnicodeDecodeError is a ValueError, not an OSError, so invalid-UTF-8
        # input would otherwise escape the loader's typed read-error contract.
        raise GoldenSetReadException(path, exc) from exc

    if not lines:
        return _malformed(row_index=1, message="Golden set is empty")
    try:
        metadata = json.loads(lines[0])
        version = metadata["version"]
        provenance_schema_version = metadata.get("provenance_schema_version")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return _malformed(row_index=1, message=f"Malformed golden-set metadata: {exc}")
    if not isinstance(version, str) or not version:
        return _malformed(row_index=1, message="Golden-set version must be a non-empty string")

    raw_queries: list[object] = []
    for row_index, line in enumerate(lines[1:], start=2):
        try:
            raw_queries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            return _malformed(row_index=row_index, message=f"Malformed JSON: {exc.msg}")
    try:
        queries = TypeAdapter(list[GoldenQuery]).validate_python(raw_queries)
    except ValidationError as exc:
        offset = exc.errors()[0]["loc"][0]
        row_index = int(offset) + 2 if isinstance(offset, int) else 2
        return _malformed(row_index=row_index, message=str(exc))
    if not queries:
        return _malformed(row_index=2, message="Golden set contains no query rows")
    return Success(
        GoldenSet(
            version=version,
            provenance_schema_version=provenance_schema_version,
            queries=queries,
        )
    )
