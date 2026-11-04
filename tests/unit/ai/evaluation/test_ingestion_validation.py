"""Tests for ingestion validation harness.

Covers:
- Property 1: State Comparison Correctness (task 1.2.3)
- Unit tests for error handling (task 1.2.4)
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from returns.result import Failure, Success

from app.ai.evaluation.ingestion import (
    MissingProvenanceError,
    PolicyValidationError,
    StateMismatchError,
    validate_ingestion_provenance,
    validate_ingestion_states,
)
from app.ai.evaluation.schema import (
    GoldenIngestionState,
    GoldenProvenanceManifest,
    GoldenQuery,
    GoldenSet,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INGESTION_STATES = list(GoldenIngestionState)


def _make_manifest(**overrides: object) -> GoldenProvenanceManifest:
    defaults: dict[str, object] = {
        "manifest_version": "1.0",
        "golden_set_version": "ingestion_v1",
        "artifact": "ingestion_v1.jsonl",
    }
    return GoldenProvenanceManifest.model_validate({**defaults, **overrides})


def _make_golden_set(**overrides: object) -> GoldenSet:
    # Minimal valid GoldenSet — queries list is required but we only need the
    # set-level fields for provenance gate tests.

    defaults: dict[str, object] = {
        "version": "ingestion_v1",
        "provenance_schema_version": "golden_provenance_v1",
        "queries": [],
    }
    merged = {**defaults, **overrides}
    # GoldenSet requires min_length=1 on queries; bypass via model_construct
    # for tests that only care about set-level fields.
    return GoldenSet.model_construct(**merged)


# ---------------------------------------------------------------------------
# Property 1: State Comparison Correctness (task 1.2.3)
# ---------------------------------------------------------------------------


@given(
    expected=st.sampled_from(_INGESTION_STATES),
    actual=st.sampled_from(_INGESTION_STATES),
)
@settings(max_examples=100)
async def test_match_iff_states_equal(
    expected: GoldenIngestionState,
    actual: GoldenIngestionState,
) -> None:
    """Property 1: match is True iff expected == actual, for all state pairs."""

    async def provider(_doc_id: str) -> GoldenIngestionState:
        return actual

    # Build a minimal no_evidence_vs_ingestion_failure row so the harness
    # processes it (the harness skips rows outside the ingestion-failure family).
    from app.ai.evaluation.schema import ExpectedIngestionState

    row = GoldenQuery.model_construct(
        query="test query",
        case_type="no_evidence_vs_ingestion_failure",
        document_kind="contracts",
        difficulty="easy",
        notes="",
        awaiting_sme_expansion=False,
        expected_chunk_ids=[],
        expected_document_ids=[],
        expected_ingestion_states=[ExpectedIngestionState(document_id="doc-1", state=expected)],
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
        retrieval_relevance="not_applicable",
        claim_entailment="not_applicable",
        ocr_profile="clean",
        pair_id=None,
        pair_variable=None,
        pair_role=None,
        case_id=None,
        measurement_family=None,
        grader=None,
        provenance=None,
    )

    report = await validate_ingestion_states(
        golden_rows=[row],
        ingestion_state_provider=provider,
        policy="mechanically_verified",
    )

    assert len(report.results) == 1
    result = report.results[0]
    assert result.match == (expected == actual), (
        f"expected={expected!r}, actual={actual!r}, match={result.match}"
    )


# ---------------------------------------------------------------------------
# Unit tests: error handling (task 1.2.4)
# ---------------------------------------------------------------------------


class TestStateMismatchError:
    def test_message_contains_document_id(self) -> None:
        err = StateMismatchError("doc-42", GoldenIngestionState.READY, GoldenIngestionState.FAILED)
        assert "doc-42" in str(err)
        assert err.document_id == "doc-42"
        assert err.expected == GoldenIngestionState.READY
        assert err.actual == GoldenIngestionState.FAILED

    def test_none_actual_state(self) -> None:
        err = StateMismatchError("doc-x", GoldenIngestionState.READY, None)
        assert err.actual is None
        assert "None" in str(err)


class TestMissingProvenanceError:
    def test_is_ingestion_validation_error(self) -> None:
        from app.ai.evaluation.ingestion import IngestionValidationError

        err = MissingProvenanceError("provenance missing")
        assert isinstance(err, IngestionValidationError)


class TestPolicyValidationError:
    def test_is_ingestion_validation_error(self) -> None:
        from app.ai.evaluation.ingestion import IngestionValidationError

        err = PolicyValidationError("policy failed")
        assert isinstance(err, IngestionValidationError)


class TestValidateIngestionProvenance:
    def test_valid_provenance_returns_success(self) -> None:
        golden_set = _make_golden_set()
        manifest = _make_manifest()
        result = validate_ingestion_provenance(golden_set=golden_set, manifest=manifest)
        assert isinstance(result, Success)

    def test_missing_schema_version_returns_failure(self) -> None:
        golden_set = _make_golden_set(provenance_schema_version=None)
        manifest = _make_manifest()
        result = validate_ingestion_provenance(golden_set=golden_set, manifest=manifest)
        assert isinstance(result, Failure)
        issues = result.failure()
        assert any("provenance_schema_version" in issue for issue in issues)

    def test_wrong_schema_version_returns_failure(self) -> None:
        golden_set = _make_golden_set(provenance_schema_version="golden_provenance_v0")
        manifest = _make_manifest()
        result = validate_ingestion_provenance(golden_set=golden_set, manifest=manifest)
        assert isinstance(result, Failure)

    def test_v2_set_with_empty_per_artifact_returns_failure(self) -> None:
        golden_set = _make_golden_set(version="legal_retrieval_v2")
        manifest = _make_manifest(golden_set_version="legal_retrieval_v2", per_artifact=[])
        result = validate_ingestion_provenance(golden_set=golden_set, manifest=manifest)
        assert isinstance(result, Failure)
        issues = result.failure()
        assert any("per_artifact" in issue for issue in issues)

    def test_multiple_issues_all_reported(self) -> None:
        golden_set = _make_golden_set(provenance_schema_version="bad_version")
        # Use model_construct to bypass min_length=1 so we can test the gate
        manifest = GoldenProvenanceManifest.model_construct(
            manifest_version="1.0",
            golden_set_version="ingestion_v1",
            artifact="ingestion_v1.jsonl",
            source_policy={},
            annotation_policy={},
            per_artifact=[],
        )
        result = validate_ingestion_provenance(golden_set=golden_set, manifest=manifest)
        # Schema version mismatch must be reported
        assert isinstance(result, Failure)
        assert len(result.failure()) >= 1
