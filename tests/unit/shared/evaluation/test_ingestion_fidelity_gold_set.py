"""Task 4.6/4.7 regression protection for the ingestion-fidelity gold set.

- The versioned ingestion gold set (``evals/golden/ingestion_fidelity_v1.jsonl``)
  validates through the provenance validator and covers every required
  I4 profile — scanned PDFs, tables, clause numbering, annexures,
  multi-contract files, and Phase-3-style clause-kind documents — with rows
  that label source-to-chunk coverage/order, never retrieval relevance.
- The published machine-readable fidelity report
  (``evals/reports/ingestion_fidelity_i4_offline.json``) passes the
  pilot-blocking coverage/order/warning gate over exactly what the offline
  deterministic stages cover, and honestly records the live-run pending state
  and the explicitly-unavailable metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from returns.result import Success

from app.ai.evaluation.ingestion_fidelity import (
    GateOutcome,
    OfflineArtifactFidelity,
    evaluate_row,
)
from app.ai.evaluation.schema import (
    INGESTION_FIDELITY_SET_VERSION,
    IngestionFidelityExpectation,
    IngestionFidelityRow,
    load_ingestion_fidelity_set,
)

GOLD_SET_PATH = Path("evals/golden/ingestion_fidelity_v1.jsonl")
REPORT_PATH = Path("evals/reports/ingestion_fidelity_i4_offline.json")


def _raw_gold_rows() -> list[dict[str, object]]:
    """The gold set's data rows (metadata line dropped), read outside async fns."""
    return [
        json.loads(line)
        for line in GOLD_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][1:]


@pytest.mark.asyncio
async def test_ingestion_fidelity_gold_set_is_valid_and_covers_i4_profiles() -> None:
    result = await load_ingestion_fidelity_set(GOLD_SET_PATH)

    assert isinstance(result, Success)
    gold_set = result.unwrap()
    assert gold_set.version == INGESTION_FIDELITY_SET_VERSION
    assert len(gold_set.rows) == 25

    profiles = {profile for row in gold_set.rows for profile in row.profiles}
    # Scanned PDFs, tables, clause numbering, multi-contract files.
    assert {"scan_heavy", "table_schedule_heavy", "irregular_numbering", "multi_contract"} <= (
        profiles
    )
    labels = {str(row.label) for row in gold_set.rows}
    assert labels == {"coverage_order", "warning_reason", "boundary_identity", "clause_kind"}
    # Annexure coverage: at least one row requires Schedule/Annex loci.
    annexure_rows = [
        row
        for row in gold_set.rows
        if any(
            prefix.startswith(("Schedule", "Annex"))
            for prefix in row.expectation.expected_locus_prefixes
        )
    ]
    assert annexure_rows, "annexure coverage must be labeled explicitly"
    # Phase-3-style clause-kind documents.
    assert any(row.label == "clause_kind" for row in gold_set.rows)

    raw_rows = _raw_gold_rows()
    forbidden = {
        "query",
        "queries",
        "retrieval_relevance",
        "claim_entailment",
        "expected_chunk_ids_relevant",
    }
    for raw in raw_rows:
        blob = json.dumps(raw)
        assert not forbidden & set(raw), (
            f"row {raw.get('case_id')} carries a retrieval-relevance field"
        )
        assert "retrieval_relevance" not in blob
        assert "claim_entailment" not in blob


def test_published_fidelity_report_passes_gate_and_records_live_pending() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["report_version"] == "ingestion_fidelity_report_v1"
    assert report["golden_set_version"] == INGESTION_FIDELITY_SET_VERSION
    gate = report["gate"]
    assert gate["verdict"] == "pass"
    assert gate["rows_failed"] == 0
    assert gate["failed_case_ids"] == []
    assert gate["rows_passed"] == 25
    # The live queued run is quota-blocked and recorded as pending, not faked.
    assert report["live_run"]["status"] == "pending"
    assert "quota" in report["live_run"]["reason"]
    assert report["limitations"]

    artifacts = report["artifacts"]
    assert len(artifacts) == 17
    for artifact in artifacts:
        # Coverage is the measured, gated dimension; order is None only when
        # no chunk could be located at all (the recorded scanned profiles with
        # an empty text layer), which the gold rows then skip honestly.
        assert artifact["source_coverage_ratio"] is not None
        if artifact["chunk_count"] == 0:
            assert artifact["source_order_preserved"] is None
        else:
            assert artifact["source_order_preserved"] is not None
        assert artifact["boundary_count"] >= 1
        # Offline honesty: OCR and chunk-count plausibility are explicitly
        # unavailable, never fabricated.
        assert artifact["ocr_status"] == "unavailable"
        assert artifact["chunk_plausibility_verdict"] == "unavailable"
        assert set(artifact["unavailable_metrics"]) >= {
            "ocr_confidence",
            "chunk_plausibility_tokenizer_bounds",
        }
    # Every gold row was evaluated against its artifact's measurement.
    evaluated = {row["case_id"] for row in report["rows"]}
    assert evaluated == {row["case_id"] for row in _raw_gold_rows()}


def _row(expectation_kwargs: dict[str, object]) -> IngestionFidelityRow:
    return IngestionFidelityRow(
        case_id="coverage_order-unit-row",
        label="coverage_order",
        document_id="UNIT",
        profiles=["clean_control"],
        expectation=IngestionFidelityExpectation(**expectation_kwargs),
        provenance={
            "source_type": "synthetic",
            "source_reference": "unit test",
            "permission_status": "synthetic_owned",
            "license_terms_status": "not_required",
            "is_public": False,
            "is_synthetic": True,
            "human_review_status": "engineer_reviewed",
            "annotators": {
                "engineer_relevance": "unit-test-labeler",
                "legal_sme_entailment": "not-required-ingestion-fidelity",
            },
        },
        notes="unit fixture",
    )


def _measurement(**overrides: object) -> OfflineArtifactFidelity:
    defaults: dict[str, object] = {
        "source_id": "UNIT",
        "pages": 1,
        "source_chars": 100,
        "chunk_count": 2,
        "first_chunk_id": "UNIT:p1:c01",
        "last_chunk_id": "UNIT:p1:c02",
        "source_coverage_ratio": 0.97,
        "source_order_preserved": True,
        "sections_total": 4,
        "sections_preserved": 4,
        "warning_codes": [],
    }
    return OfflineArtifactFidelity.model_validate({**defaults, **overrides})


def test_gate_row_evaluation_blocks_on_failed_coverage_expectation() -> None:
    passed = evaluate_row(
        _row({"min_source_coverage": 0.95, "order_preserved": True}), _measurement()
    )
    assert passed.passed

    blocked = evaluate_row(
        _row({"min_source_coverage": 0.99, "order_preserved": True}), _measurement()
    )
    assert not blocked.passed
    assert blocked.failures
    assert "source coverage" in blocked.failures[0]

    order_violation = evaluate_row(
        _row({"min_source_coverage": 0.95, "order_preserved": True}),
        _measurement(source_order_preserved=False),
    )
    assert not order_violation.passed
    assert any("order" in failure for failure in order_violation.failures)


def test_gate_row_evaluation_blocks_on_unexpected_and_missing_warnings() -> None:
    missing = evaluate_row(
        _row({"expected_warning_codes": ["FIDELITY_OCR_CONFIDENCE_LOW"]}), _measurement()
    )
    assert not missing.passed
    assert any("did not fire" in failure for failure in missing.failures)

    fired = evaluate_row(
        _row({"forbidden_warning_codes": ["FIDELITY_TABLE_LOSS"]}),
        _measurement(warning_codes=["FIDELITY_TABLE_LOSS"]),
    )
    assert not fired.passed
    assert any("fired" in failure for failure in fired.failures)

    # A row whose offline-unevaluable dimensions are the only expectations is
    # skipped — recorded, never silently passed.
    skipped = evaluate_row(_row({"max_ocr_confidence": 0.5}), _measurement())
    assert not skipped.passed
    assert skipped.skipped


def test_gate_outcome_reflects_failed_rows() -> None:
    gate = GateOutcome(
        verdict="fail",
        basis="labeled source-to-chunk coverage/order and typed warning outcomes",
        rows_passed=24,
        rows_failed=1,
        failed_case_ids=["coverage_order-SRC-01"],
    )
    assert gate.verdict == "fail"
    assert gate.failed_case_ids == ["coverage_order-SRC-01"]
