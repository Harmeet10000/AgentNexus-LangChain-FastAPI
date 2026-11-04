"""Property tests for ingestion coverage metrics.

Property 2: Coverage Metrics Bounded [0, 1] (task 1.3.3)
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.ai.evaluation.ingestion import audit_ingestion_coverage, generate_candidate_queries
from app.ai.evaluation.schema import CoverageGap, GoldenIngestionState, GoldenQuery

pytestmark = pytest.mark.property

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_INGESTION_CASE_TYPES = st.sampled_from(
    [
        "no_evidence_vs_ingestion_failure",
        "index_not_ready",
        "partial_coverage_answer",
        "direct_answer",
        "multi_hop",
    ]
)


_INGESTION_FAILURE_TYPES = {"no_evidence_vs_ingestion_failure", "index_not_ready"}


def _minimal_row(case_type: str) -> GoldenQuery:
    """Build the minimal GoldenQuery model_construct for a given case_type."""
    from app.ai.evaluation.schema import ExpectedIngestionState

    ingestion_states = (
        [ExpectedIngestionState(document_id="doc-1", state=GoldenIngestionState.READY)]
        if case_type in _INGESTION_FAILURE_TYPES
        else []
    )
    return GoldenQuery.model_construct(
        query="test query for coverage",
        case_type=case_type,
        document_kind="contracts",
        difficulty="easy",
        notes="",
        awaiting_sme_expansion=False,
        expected_chunk_ids=[] if case_type in _INGESTION_FAILURE_TYPES else ["chunk-1"],
        expected_document_ids=[],
        expected_ingestion_states=ingestion_states,
        expected_evidence_policy="mechanically_verified",
        policy_overrides={},
        expected_claims=[],
        expected_spans=[],
        expected_contract_indexes=[],
        expected_document_versions=[],
        stale_document_ids=[],
        forbidden_document_ids=[],
        forbidden_chunk_ids=[],
        forbidden_contract_indexes=[],
        allowed_doc_ids=[],
        expected_answer_must_include=[],
        expected_answer_must_not_include=[],
        adversarial_markers=[],
        adversarial=False,
        expected_abstention=False,
        expected_evidence_status=None,
        expected_coverage=None,
        expected_coverage_ratio_min=None,
        retrieval_relevance="not_applicable"
        if case_type in _INGESTION_FAILURE_TYPES
        else "relevant",
        claim_entailment="not_applicable" if case_type in _INGESTION_FAILURE_TYPES else "entailed",
        ocr_profile="clean",
        pair_id=None,
        pair_variable=None,
        pair_role=None,
        case_id=None,
        measurement_family=None,
        grader=None,
        provenance=None,
    )


# ---------------------------------------------------------------------------
# Property 2: Coverage Metrics Bounded [0, 1]
# ---------------------------------------------------------------------------


@given(
    case_types=st.lists(_INGESTION_CASE_TYPES, min_size=1, max_size=20),
    minimum=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
async def test_coverage_gap_ratio_bounded(
    case_types: list[str],
    minimum: int,
) -> None:
    """Property 2: gap_ratio for every CoverageGap is in [0, 1]."""
    rows = [_minimal_row(ct) for ct in case_types]
    report = await audit_ingestion_coverage(golden_rows=rows, minimum_per_type=minimum)

    for gap in report.gaps:
        ratio = gap.current_count / gap.required_minimum
        assert 0.0 <= ratio <= 1.0, (
            f"gap_ratio out of bounds for {gap.case_type}: "
            f"{gap.current_count}/{gap.required_minimum} = {ratio}"
        )


@given(
    case_types=st.lists(_INGESTION_CASE_TYPES, min_size=1, max_size=20),
    minimum=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
async def test_coverage_total_rows_non_negative(
    case_types: list[str],
    minimum: int,
) -> None:
    """Property 2 (auxiliary): total_rows equals len(input rows)."""
    rows = [_minimal_row(ct) for ct in case_types]
    report = await audit_ingestion_coverage(golden_rows=rows, minimum_per_type=minimum)
    assert report.total_rows == len(rows)
    assert report.total_rows >= 0


@given(
    case_types=st.lists(_INGESTION_CASE_TYPES, min_size=1, max_size=20),
    minimum=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
async def test_case_type_counts_sum_to_total(
    case_types: list[str],
    minimum: int,
) -> None:
    """Property 2 (auxiliary): sum of per-type counts equals total_rows."""
    rows = [_minimal_row(ct) for ct in case_types]
    report = await audit_ingestion_coverage(golden_rows=rows, minimum_per_type=minimum)
    assert sum(report.case_type_coverage.values()) == report.total_rows


# ---------------------------------------------------------------------------
# Task 1.3.2: candidate query generation
# ---------------------------------------------------------------------------


class TestGenerateCandidateQueries:
    def test_returns_empty_for_no_gaps(self) -> None:
        assert generate_candidate_queries([]) == []

    def test_generates_up_to_needed_count(self) -> None:
        gap = CoverageGap(
            case_type="no_evidence_vs_ingestion_failure",
            current_count=2,
            required_minimum=5,
            suggested_queries=[],
        )
        candidates = generate_candidate_queries([gap])
        # needed = 5 - 2 = 3; template list has 5 entries
        assert len(candidates) == 3

    def test_candidate_has_required_keys(self) -> None:
        gap = CoverageGap(
            case_type="index_not_ready",
            current_count=0,
            required_minimum=5,
            suggested_queries=[],
        )
        candidates = generate_candidate_queries([gap])
        for c in candidates:
            assert "query" in c
            assert "case_type" in c
            assert "notes" in c
            assert c["case_type"] == "index_not_ready"

    def test_unknown_case_type_returns_empty(self) -> None:
        gap = CoverageGap(
            case_type="ocr_semantic_flip",
            current_count=0,
            required_minimum=5,
            suggested_queries=[],
        )
        candidates = generate_candidate_queries([gap])
        assert candidates == []
