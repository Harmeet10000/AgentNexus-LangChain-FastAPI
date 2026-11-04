"""Tests for feeding taxonomy candidates into golden-set rows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.ai.evaluation.golden_set_validator import GoldenSetValidationError
from app.ai.evaluation.schema import (
    ExpectedSpan,
    GoldenQuery,
    TaxonomyCandidate,
)
from app.ai.evaluation.signal_ingestion import SupportTicket, TaxonomyCandidateGenerator
from app.ai.evaluation.taxonomy_feeder import DefaultTaxonomyCandidateFeeder

pytestmark = pytest.mark.unit


def _candidate(
    *,
    case_type: str = "unanswerable",
    query: str = "Why can't I locate the indemnification clause?",
) -> TaxonomyCandidate:
    return TaxonomyCandidate(
        query_template=query,
        recommended_case_type=case_type,
        source_ticket_ids=["T-1", "T-2", "T-3"],
        cluster_similarity_score=0.91,
        symptom_variations=["cannot find clause", "clause missing"],
    )


def _answerable_row() -> GoldenQuery:
    return GoldenQuery(
        query="Where is the indemnification clause?",
        expected_chunk_ids=["chunk-1", "chunk-2"],
        expected_document_ids=["SRC-1"],
        document_kind="contracts",
        difficulty="hard",
        notes="curated",
        awaiting_sme_expansion=True,
        case_type="direct_answer",
        expected_spans=[
            ExpectedSpan(document_id="SRC-1", page_no=2, quote="indemnification clause")
        ],
    )


def _ticket(index: int, symptom: str) -> SupportTicket:
    return SupportTicket(
        ticket_id=f"T-{index}",
        customer_id=f"customer-{index}",
        issue_category="retrieval_failure",
        symptom_description=symptom,
        resolution_note=None,
        created_at=(datetime(2024, 3, 1, tzinfo=UTC) + timedelta(days=index))
        .isoformat()
        .replace("+00:00", "Z"),
        resolved_at=None,
    )


@pytest.mark.property
@pytest.mark.parametrize(
    "symptoms",
    [
        ["Cannot locate indemnification provisions in MSA"],
        ["Indemnity clause search returns empty results", "Search finds nothing"],
        ["Looking for payment terms but getting no matches"],
    ],
)
def test_property_9_candidate_validation(symptoms: list[str]) -> None:
    """Feature: evaluation-infrastructure-extension, Property 9."""
    tickets = [
        _ticket(index, symptoms[index % len(symptoms)]) for index in range(len(symptoms) * 5)
    ]
    candidates = TaxonomyCandidateGenerator().generate_candidates(
        {"cluster-1": tickets}, threshold=5
    )

    assert candidates
    for candidate in candidates:
        assert candidate.query_template.strip()
        assert candidate.recommended_case_type in {
            "unanswerable",
            "no_evidence_vs_ingestion_failure",
        }
        assert candidate.source_ticket_ids


def test_conversion_preserves_expected_evidence_from_source() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()
    source = _answerable_row()

    row = feeder.to_golden_row(
        _candidate(case_type="direct_answer"),
        source_golden_row=source,
    )

    assert row.query == _candidate(case_type="direct_answer").query_template
    assert row.expected_chunk_ids == source.expected_chunk_ids
    assert row.expected_document_ids == source.expected_document_ids
    assert row.expected_spans == source.expected_spans
    assert row.document_kind == "contracts"
    assert row.difficulty == "hard"
    assert row.awaiting_sme_expansion is False
    assert all(ticket in row.notes for ticket in ("T-1", "T-2", "T-3"))
    assert row.provenance is not None
    assert row.provenance.quality_tier == "candidate"


def test_no_evidence_candidate_drops_expected_evidence() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()
    source = _answerable_row()

    row = feeder.to_golden_row(_candidate(case_type="unanswerable"), source_golden_row=source)

    assert row.case_type == "unanswerable"
    assert row.expected_chunk_ids == []
    assert row.expected_document_ids == []
    assert row.expected_spans == []
    assert row.retrieval_relevance == "not_applicable"
    assert row.claim_entailment == "not_applicable"


def test_answerable_candidate_without_evidence_raises() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()

    with pytest.raises(GoldenSetValidationError):
        feeder.to_golden_row(_candidate(case_type="direct_answer"))


def test_invalid_recommended_case_type_raises() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()

    with pytest.raises(GoldenSetValidationError):
        feeder.to_golden_row(_candidate(case_type="not_a_case_type"))


def test_ab_pair_shape_is_valid_and_differs_only_by_variable() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()
    base = _answerable_row()

    control, treatment = feeder.generate_pair_for_ab_test(
        base,
        "query",
        "Why can't I find the indemnification clause?",
    )

    assert control.pair_id == treatment.pair_id
    assert control.pair_variable == treatment.pair_variable == "query"
    assert control.pair_role == "control"
    assert treatment.pair_role == "treatment"
    assert control.query == base.query
    assert treatment.query == "Why can't I find the indemnification clause?"
    assert control.model_dump(exclude={"pair_role", "query", "case_id"}) == treatment.model_dump(
        exclude={"pair_role", "query", "case_id"}
    )


def test_ab_pair_rejects_unknown_and_pair_metadata_variables() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()
    base = _answerable_row()

    with pytest.raises(GoldenSetValidationError):
        feeder.generate_pair_for_ab_test(base, "not_a_field", "value")
    with pytest.raises(GoldenSetValidationError):
        feeder.generate_pair_for_ab_test(base, "pair_id", "value")
    with pytest.raises(GoldenSetValidationError):
        feeder.generate_pair_for_ab_test(base, "pair_role", "treatment")


def test_validate_before_commit_accepts_valid_rows() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()

    feeder.validate_before_commit([_answerable_row()])


def test_validate_before_commit_rejects_invalid_rows() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()
    invalid = _answerable_row().model_copy(update={"query": ""})

    with pytest.raises((GoldenSetValidationError, ValidationError)):
        feeder.validate_before_commit([invalid])


def test_ab_pair_rows_validate_together_without_duplicate_ids() -> None:
    """The pair must pass the set-level gate with distinct case_ids."""
    from app.ai.evaluation.schema import (
        AnnotatorRoles,
        GoldenItemProvenance,
        GoldenSet,
        validate_golden_set,
    )

    feeder = DefaultTaxonomyCandidateFeeder()
    source = _answerable_row().model_copy(
        update={
            "case_id": "direct_answer-feed-x-001",
            "provenance": GoldenItemProvenance(
                source_type="synthetic",
                source_reference="unit-test fixture",
                permission_status="synthetic_owned",
                license_terms_status="not_required",
                is_public=False,
                is_synthetic=True,
                human_review_status="pending",
                annotators=AnnotatorRoles(
                    engineer_relevance="unit-test",
                    legal_sme_entailment="sme-pending",
                ),
                quality_tier="candidate",
            ),
        }
    )
    control, treatment = feeder.generate_pair_for_ab_test(
        source, "query", "Where is the indemnification clause?"
    )

    assert control.case_id != treatment.case_id
    assert control.case_id.startswith(f"{control.case_type}-pair-")
    assert treatment.case_id.startswith(f"{treatment.case_type}-pair-")

    validate_golden_set(
        GoldenSet(
            version="ingestion_v1",
            provenance_schema_version="golden_provenance_v1",
            queries=[control, treatment],
        )
    )


def test_ab_pair_type_change_keeps_valid_case_id_prefixes() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()
    source = _answerable_row().model_copy(update={"case_id": "direct_answer-prefix-001"})

    control, treatment = feeder.generate_pair_for_ab_test(source, "case_type", "multi_document")

    assert control.case_id.startswith("direct_answer-pair-")
    assert treatment.case_type == "multi_document"
    assert treatment.case_id.startswith("multi_document-pair-")


def test_ab_pairs_over_same_base_row_are_distinct() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()
    source = _answerable_row().model_copy(update={"case_id": "direct_answer-multi-001"})

    first = feeder.generate_pair_for_ab_test(source, "query", "variant A")
    second = feeder.generate_pair_for_ab_test(source, "query", "variant B")

    assert first[0].pair_id != second[0].pair_id
    assert first[0].case_id != second[0].case_id


def test_ab_pair_rejects_case_id_variable() -> None:
    feeder = DefaultTaxonomyCandidateFeeder()

    with pytest.raises(GoldenSetValidationError, match="case_id"):
        feeder.generate_pair_for_ab_test(_answerable_row(), "case_id", "my-anything")
