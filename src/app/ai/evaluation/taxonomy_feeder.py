"""Taxonomy candidate feeding for converting candidates to golden set rows."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, cast, get_args

from pydantic import ValidationError

from .golden_set_validator import GoldenSetValidationError
from .schema import (
    COVERAGE_CASE_TYPES,
    NO_EVIDENCE_CASE_TYPES,
    AnnotatorRoles,
    GoldenCaseType,
    GoldenItemProvenance,
    GoldenQuery,
)

if TYPE_CHECKING:
    from .schema import TaxonomyCandidate

_VALID_CASE_TYPES: frozenset[str] = frozenset(get_args(GoldenCaseType.__value__))
_PAIR_METADATA_FIELDS = frozenset({"pair_id", "pair_variable", "pair_role"})


class TaxonomyCandidateFeeder(Protocol):
    """Convert taxonomy candidates to golden set rows.

    This interface defines the contract for feeding approved taxonomy candidates
    into the golden set, supporting both single row conversion and A/B testing pairs.
    """

    def to_golden_row(
        self,
        candidate: TaxonomyCandidate,
        source_golden_row: GoldenQuery | None = None,
    ) -> GoldenQuery:
        """Convert a taxonomy candidate to a golden set row."""
        ...

    def generate_pair_for_ab_test(
        self,
        base_row: GoldenQuery,
        variable_name: str,
        treatment_value: str,
    ) -> tuple[GoldenQuery, GoldenQuery]:
        """Generate a control-treatment pair for A/B testing."""
        ...


class DefaultTaxonomyCandidateFeeder:
    """Default candidate feeder backed by the golden-set schema contracts."""

    def to_golden_row(
        self,
        candidate: TaxonomyCandidate,
        source_golden_row: GoldenQuery | None = None,
    ) -> GoldenQuery:
        """Convert a candidate into a validated golden-set row."""
        case_type_name = candidate.recommended_case_type
        if case_type_name not in _VALID_CASE_TYPES:
            message = f"recommended_case_type is not a valid GoldenCaseType: {case_type_name!r}"
            raise GoldenSetValidationError(message)
        case_type = cast("GoldenCaseType", case_type_name)
        no_evidence = case_type in NO_EVIDENCE_CASE_TYPES
        # Every candidate gets its own identity: the source row's case_id would
        # duplicate the set and break the case-type slug prefix when the
        # recommended case type differs. Deterministic from the candidate so
        # reruns are stable.
        candidate_material = "\0".join([candidate.query_template, *candidate.source_ticket_ids])
        candidate_case_id = (
            f"{case_type}-candidate-{hashlib.sha256(candidate_material.encode()).hexdigest()[:16]}"
        )

        # Source rows carry the SME-curated expectation labels. Where the candidate is
        # derived from one, those labels ride along; where they are required and missing,
        # the error names the missing label instead of constructing a row that would fail
        # the golden-set contract later and less clearly.
        preserved = (
            source_golden_row.model_dump(
                include={
                    "expected_document_ids",
                    "expected_chunk_ids",
                    "expected_spans",
                    "retrieval_relevance",
                    "claim_entailment",
                    "jurisdiction",
                    "expected_coverage",
                    "expected_coverage_ratio_min",
                    "expected_evidence_status",
                    "expected_abstention",
                    "expected_ingestion_states",
                    "policy_overrides",
                    "expected_evidence_policy",
                    "measurement_family",
                    "expected_claims",
                    "expected_answer_must_include",
                    "expected_answer_must_not_include",
                    "stale_document_ids",
                    "forbidden_document_ids",
                    "forbidden_chunk_ids",
                    "expected_contract_indexes",
                    "expected_document_versions",
                    "grader",
                    "ocr_profile",
                    "adversarial",
                    "adversarial_markers",
                },
                mode="json",
            )
            if source_golden_row is not None
            else {}
        )
        coverage_expectation = preserved.get("expected_coverage") or preserved.get(
            "expected_coverage_ratio_min"
        )

        if no_evidence:
            preserved["expected_chunk_ids"] = []
            preserved["expected_document_ids"] = []
            preserved["expected_spans"] = []
            preserved["expected_claims"] = []
            preserved["expected_answer_must_include"] = []
            preserved["expected_answer_must_not_include"] = []
            preserved["retrieval_relevance"] = "not_applicable"
            preserved["claim_entailment"] = "not_applicable"
            if coverage_expectation is None and case_type in COVERAGE_CASE_TYPES:
                message = (
                    f"{case_type} candidates need a coverage expectation from the source "
                    "row (expected_coverage or expected_coverage_ratio_min)"
                )
                raise GoldenSetValidationError(message)
        else:
            if not preserved.get("expected_chunk_ids"):
                message = (
                    f"{case_type} candidates require expected_chunk_ids from a source golden row"
                )
                raise GoldenSetValidationError(message)
            if case_type in COVERAGE_CASE_TYPES and coverage_expectation is None:
                message = (
                    f"{case_type} candidates need a coverage expectation from the source "
                    "row (expected_coverage or expected_coverage_ratio_min)"
                )
                raise GoldenSetValidationError(message)

        payload = {
            "query": candidate.query_template,
            "case_id": candidate_case_id,
            "document_kind": source_golden_row.document_kind if source_golden_row else "contracts",
            "difficulty": source_golden_row.difficulty if source_golden_row else "medium",
            "notes": (
                f"Generated from taxonomy candidate with "
                f"{len(candidate.source_ticket_ids)} source tickets: "
                f"{', '.join(candidate.source_ticket_ids)}; cluster similarity "
                f"{candidate.cluster_similarity_score:.3f}."
            ),
            "awaiting_sme_expansion": False,
            "case_type": case_type,
            "provenance": self._build_provenance(candidate, source_golden_row),
        }
        payload.update(preserved)
        try:
            return GoldenQuery.model_validate(payload)
        except ValidationError as exc:
            message = f"candidate produced an invalid golden row: {exc}"
            raise GoldenSetValidationError(message) from exc

    def generate_pair_for_ab_test(  # noqa: PLR6301
        self,
        base_row: GoldenQuery,
        variable_name: str,
        treatment_value: str,
    ) -> tuple[GoldenQuery, GoldenQuery]:
        """Return control/treatment rows differing only in the tested variable."""
        if variable_name not in GoldenQuery.model_fields:
            message = f"unknown GoldenQuery field: {variable_name!r}"
            raise GoldenSetValidationError(message)
        if variable_name in _PAIR_METADATA_FIELDS:
            message = f"cannot A/B test pair metadata field: {variable_name!r}"
            raise GoldenSetValidationError(message)

        pair_id = hashlib.sha256(f"{base_row.query}\0{variable_name}".encode()).hexdigest()[:16]
        pair_id = f"pair-{pair_id}"
        # Pair rows are new rows and must carry their own distinct ids, each
        # with a slug prefix matching its own final case_type (case types can
        # change under A/B variable assignment), and no two pairs sharing one.
        pair_stem = pair_id.removeprefix("pair-")

        def pair_case_id(payload: dict[str, object], role: str) -> str:
            return f"{payload['case_type']}-pair-{pair_stem}-{role}"

        control_payload = base_row.model_dump()
        control_payload.update(
            {
                "pair_id": pair_id,
                "pair_variable": variable_name,
                "pair_role": "control",
            }
        )
        treatment_payload = {
            **control_payload,
            variable_name: treatment_value,
            "pair_role": "treatment",
        }
        control_payload["case_id"] = pair_case_id(control_payload, "control")
        treatment_payload["case_id"] = pair_case_id(treatment_payload, "treatment")
        try:
            control = GoldenQuery.model_validate(control_payload)
            treatment = GoldenQuery.model_validate(treatment_payload)
        except ValidationError as exc:
            message = f"invalid A/B pair: {exc}"
            raise GoldenSetValidationError(message) from exc
        return control, treatment

    def validate_before_commit(self, rows: list[GoldenQuery]) -> None:  # noqa: PLR6301
        """Re-validate rows against the golden-set schema before commit."""
        if not rows:
            message = "no golden rows supplied for validation"
            raise GoldenSetValidationError(message)
        for index, row in enumerate(rows):
            try:
                GoldenQuery.model_validate(row.model_dump())
            except ValidationError as exc:
                message = f"golden row {index} is invalid: {exc}"
                raise GoldenSetValidationError(message) from exc

    @staticmethod
    def _build_provenance(
        candidate: TaxonomyCandidate,
        source_golden_row: GoldenQuery | None,
    ) -> GoldenItemProvenance:
        if source_golden_row is not None and source_golden_row.provenance is not None:
            return source_golden_row.provenance.model_copy(update={"quality_tier": "candidate"})
        return GoldenItemProvenance(
            source_type="synthetic",
            source_reference=f"tickets:{', '.join(candidate.source_ticket_ids)}",
            permission_status="synthetic_owned",
            license_terms_status="not_required",
            is_public=False,
            is_synthetic=True,
            human_review_status="pending",
            annotators=AnnotatorRoles(
                engineer_relevance="taxonomy-candidate-generator",
                legal_sme_entailment="sme-expansion-pending",
            ),
            quality_tier="candidate",
        )
