"""Offline tests for the metric assertion gate (task 8.8)."""

from __future__ import annotations

import pytest

from app.ai.evaluation.assertions import (
    AssertionThresholds,
    collect_metrics,
    evaluate_assertions,
    threshold_provenance,
)

pytestmark = pytest.mark.unit


def _thresholds(**overrides: object) -> AssertionThresholds:
    defaults: dict[str, object] = {
        "source_report": "test-report.json",
        "golden_set_version": "answer_golden_v1",
        "recall_at_k": 0.57,
        "citation_validity": 1.0,
    }
    defaults.update(overrides)
    return AssertionThresholds.model_validate(defaults)


def test_collect_metrics_flattens_retrieval_and_entailment_aggregates() -> None:
    metrics = collect_metrics(
        retrieval_report={"aggregates": {"recall_at_k": 0.9, "ndcg_at_k": 0.5}, "rows": [1, 2]},
        entailment_report={
            "aggregates": {
                "atomic_entailment_rate": 0.5,
                "citation_validity": 1.0,
                "unsupported_claim_rate": 0.1,
                "abstention_correctness": None,
            }
        },
    )
    assert metrics["recall_at_k"] == pytest.approx(0.9)
    assert metrics["retrieval_rows"] == pytest.approx(2.0)
    assert metrics["atomic_entailment_rate"] == pytest.approx(0.5)
    assert metrics["citation_validity"] == pytest.approx(1.0)
    # Security counters default to zero (no observed violation) when a report
    # carries no such counter.
    assert metrics["scope_violations"] == pytest.approx(0.0)
    assert metrics["unauthorized_reads"] == pytest.approx(0.0)
    # The offline path leaves abstention correctness unmeasured (None) — it
    # must NOT be fabricated into the metric namespace.
    assert "abstention_correctness" not in metrics


def test_quality_floor_failure_names_metric_value_and_threshold() -> None:
    metrics = collect_metrics(
        retrieval_report={"aggregates": {"recall_at_k": 0.4}},
        entailment_report={"aggregates": {"citation_validity": 0.9}},
    )
    failures = evaluate_assertions(metrics, _thresholds())
    assert any(
        "recall_at_k" in failure and "0.4" in failure and "0.57" in failure for failure in failures
    )
    assert any("citation_validity" in failure for failure in failures)


def test_ceiling_metrics_fail_when_exceeded() -> None:
    metrics = {"p95_latency_ms": 2500.0, "total_tokens": 100.0, "total_cost_usd": 0.5}
    thresholds = _thresholds(
        p95_latency_ms=2000.0,
        total_tokens=50,
        total_cost_usd=0.25,
        recall_at_k=None,
        citation_validity=None,
    )
    failures = evaluate_assertions(metrics, thresholds)
    assert len(failures) == 3
    assert all("ceiling" in failure for failure in failures)


def test_zero_tolerance_security_metrics_fail_at_any_nonzero_count() -> None:
    metrics = {"scope_violations": 1.0, "unauthorized_reads": 2.0, "poisoning_events": 1.0}
    failures = evaluate_assertions(metrics, _thresholds(recall_at_k=None, citation_validity=None))
    assert len(failures) == 3
    assert all("zero-tolerance" in failure for failure in failures)


def test_poisoning_events_survive_the_full_collection_to_assertion_path() -> None:
    """An observed poisoning counter must reach the gate, not be dropped.

    Greptile wave-3 finding: ``collect_metrics`` copied only
    ``scope_violations``/``unauthorized_reads``, so an aggregate carrying
    ``poisoning_events: 1`` lost that value and the advertised zero-tolerance
    gate passed a poisoned run. The same copy runs for every supported report
    family, and an observed value always beats the absent-counter default.
    """
    from_entailment = collect_metrics(entailment_report={"aggregates": {"poisoning_events": 1}})
    assert from_entailment["poisoning_events"] == pytest.approx(1.0)
    failures = evaluate_assertions(from_entailment, _thresholds())
    assert any("poisoning_events" in failure for failure in failures)

    from_process = collect_metrics(process_report={"poisoning_events": 2})
    assert from_process["poisoning_events"] == pytest.approx(2.0)
    assert evaluate_assertions(from_process, _thresholds()) != []

    from_retrieval = collect_metrics(
        retrieval_report={"aggregates": {"poisoning_events": 1, "recall_at_k": 0.9}}
    )
    assert from_retrieval["poisoning_events"] == pytest.approx(1.0)
    assert any("poisoning_events" in f for f in evaluate_assertions(from_retrieval, _thresholds()))

    # An observed value from a later report family overwrites an earlier
    # family's absent-counter default — order never silences an observation.
    both = collect_metrics(
        retrieval_report={"aggregates": {}},
        entailment_report={"aggregates": {"poisoning_events": 1}},
    )
    assert both["poisoning_events"] == pytest.approx(1.0)


def test_absent_poisoning_counter_defaults_to_zero_and_gate_passes() -> None:
    metrics = collect_metrics(
        retrieval_report={"aggregates": {"recall_at_k": 0.9}},
        entailment_report={"aggregates": {"citation_validity": 1.0}},
    )
    assert metrics["poisoning_events"] == pytest.approx(0.0)
    assert metrics["scope_violations"] == pytest.approx(0.0)
    assert metrics["unauthorized_reads"] == pytest.approx(0.0)
    assert evaluate_assertions(metrics, _thresholds()) == []


def test_gated_metric_missing_from_reports_fails_closed() -> None:
    failures = evaluate_assertions({}, _thresholds())
    assert any("missing from the reports" in failure for failure in failures)


def test_ingestion_fidelity_gate_passes_and_fails() -> None:
    passing = collect_metrics(ingestion_report={"gate": {"rows_passed": 25, "rows_failed": 0}})
    assert (
        evaluate_assertions(
            passing,
            _thresholds(
                recall_at_k=None, citation_validity=None, ingestion_fidelity_rows_passed=25
            ),
        )
        == []
    )
    failing = collect_metrics(ingestion_report={"gate": {"rows_passed": 24, "rows_failed": 1}})
    failures = evaluate_assertions(
        failing,
        _thresholds(recall_at_k=None, citation_validity=None, ingestion_fidelity_rows_passed=25),
    )
    assert any("ingestion_fidelity" in failure for failure in failures)


def test_threshold_provenance_cites_source_report() -> None:
    provenance = threshold_provenance(_thresholds())
    assert "test-report.json" in provenance
    assert "answer_golden_v1" in provenance


def test_recorded_assertion_thresholds_file_validates() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    thresholds_path = repo_root / "evals/reports/answer_golden_v1.assertions.json"
    thresholds = AssertionThresholds.model_validate_json(
        thresholds_path.read_text(encoding="utf-8")
    )
    assert thresholds.citation_validity == pytest.approx(1.0)
    assert thresholds.scope_violations == 0
    assert thresholds.unauthorized_reads == 0
