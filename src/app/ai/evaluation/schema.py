"""Validated schema and loader for versioned retrieval golden sets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import aiofiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from returns.result import Failure, Result, Success

from app.utils.exceptions import InfrastructureException, NotFoundException

from .exceptions import MalformedGoldenRowError

if TYPE_CHECKING:
    from pathlib import Path


type GoldenDocumentKind = Literal["contracts", "statutes", "judgments", "filings"]
type GoldenCaseType = Literal["standard", "unanswerable", "statute_not_in_corpus", "near_miss"]
type RetrievalRelevance = Literal["relevant", "partially_relevant", "irrelevant", "not_applicable"]
type ClaimEntailment = Literal[
    "entailed", "partial", "contradicted", "unsupported", "not_applicable"
]
type ProvenanceSourceType = Literal["public", "synthetic", "client"]
type PermissionStatus = Literal["public_terms_verified", "synthetic_owned", "written_permission"]
type TermsStatus = Literal["verified", "not_required", "pending"]
type HumanReviewStatus = Literal["pending", "engineer_reviewed", "sme_reviewed", "adjudicated"]

_ADJUDICATOR_REQUIRED = "A completed disagreement requires an adjudicator"
_SOURCE_FLAGS_MISMATCH = "Public/synthetic indicators do not match source_type"
_PERMISSION_MISMATCH = "permission_status is incompatible with source_type"
_TERMS_REFERENCE_REQUIRED = "Verified license/terms status requires a reference"
_CLIENT_TERMS_REQUIRED = "Client material requires verified license/terms status"
_PUBLIC_TERMS_PENDING = "Public material with unreviewed terms is not admissible"
_ANSWERABLE_CHUNKS_REQUIRED = "Answerable and near-miss cases need expected chunk identifiers"
_NO_EVIDENCE_RELEVANCE = "No-evidence cases must use not_applicable retrieval relevance"
_NO_EVIDENCE_ENTAILMENT = "No-evidence cases must use not_applicable claim entailment"


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
    case_type: GoldenCaseType = "standard"
    retrieval_relevance: RetrievalRelevance = "relevant"
    claim_entailment: ClaimEntailment = "entailed"
    provenance: GoldenItemProvenance | None = None

    @model_validator(mode="after")
    def validate_case_expectations(self) -> GoldenQuery:
        no_evidence_case = self.case_type in {"unanswerable", "statute_not_in_corpus"}
        if no_evidence_case and (self.expected_chunk_ids or self.expected_document_ids):
            message = f"{self.case_type} cases cannot declare expected evidence"
            raise ValueError(message)
        if not no_evidence_case and not self.expected_chunk_ids:
            raise ValueError(_ANSWERABLE_CHUNKS_REQUIRED)
        if no_evidence_case and self.retrieval_relevance != "not_applicable":
            raise ValueError(_NO_EVIDENCE_RELEVANCE)
        if no_evidence_case and self.claim_entailment != "not_applicable":
            raise ValueError(_NO_EVIDENCE_ENTAILMENT)
        return self


class GoldenSet(BaseModel):
    """A comparable, explicitly versioned collection of golden queries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    queries: list[GoldenQuery] = Field(min_length=1)
    provenance_schema_version: str | None = None


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


def _malformed(*, row_index: int, message: str) -> Failure[MalformedGoldenRowError]:
    return Failure(
        MalformedGoldenRowError(
            message=message,
            row_index=row_index,
            details={"row_index": row_index},
            source="golden_set_loader",
        )
    )


def validate_golden_set(golden_set: GoldenSet) -> None:
    """Enforce provenance policy for CI and evaluation execution.

    ``load_golden_set`` remains a schema loader for backwards-compatible
    programmatic callers. This explicit gate prevents legacy rows without a
    provenance record from being silently treated as approved evaluation data.
    """
    issues: list[str] = []
    if golden_set.provenance_schema_version != "golden_provenance_v1":
        issues.append("golden set is missing provenance_schema_version=golden_provenance_v1")
    for row_index, query in enumerate(golden_set.queries, start=2):
        provenance = query.provenance
        if provenance is None:
            issues.append(f"row {row_index}: provenance is missing and needs review")
            continue
        if provenance.permission_status not in {
            "public_terms_verified",
            "synthetic_owned",
            "written_permission",
        }:
            issues.append(f"row {row_index}: permission status is not admissible")
        if provenance.license_terms_status == "pending":
            issues.append(f"row {row_index}: license/terms review is pending")
        if (
            provenance.source_type == "client"
            and provenance.permission_status != "written_permission"
        ):
            issues.append(f"row {row_index}: client material lacks written permission")
        if (
            provenance.human_review_status == "sme_reviewed"
            and provenance.annotators.legal_sme_entailment in {"", "pending"}
        ):
            issues.append(f"row {row_index}: SME-reviewed item has no legal SME annotator")
        if (
            query.case_type in {"unanswerable", "statute_not_in_corpus"}
            and query.expected_chunk_ids
        ):
            issues.append(f"row {row_index}: no-evidence case has expected chunks")
    if issues:
        raise GoldenSetProvenanceError(issues)


async def load_validated_golden_set(path: Path) -> GoldenSetLoadResult:
    """Load a golden set and fail if its provenance is not CI-admissible."""
    result = await load_golden_set(path)
    if isinstance(result, Failure):
        return result
    try:
        validate_golden_set(result.unwrap())
    except GoldenSetProvenanceError as exc:
        first_issue = exc.issues[0] if exc.issues else "Invalid golden-set provenance"
        return _malformed(row_index=1, message=first_issue)
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
