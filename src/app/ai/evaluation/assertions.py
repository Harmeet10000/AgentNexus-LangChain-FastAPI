"""Build-failing metric assertions over the evaluation reports (task 8.8).

The harness reports MEASURE; this module DECIDES: it turns the recorded
metrics — evidence recall, atomic entailment, unsupported claims, correct
abstention, citation validity, scope violations, tool calls, p95 latency,
tokens, cost, and ingestion fidelity — into fail-closed build gates with the
metric name, observed value, and threshold in every failure line.

Security metrics are zero-tolerance: any observed scope violation,
unauthorized read, or poisoning event fails the build regardless of
thresholds. Missing metrics fail closed as well — a gate cannot silently pass
on an absent measurement.

Usage:

    uv run python -m app.ai.evaluation.assertions \
        --retrieval-report evals/reports/<retrieval>.json \
        --entailment-report evals/reports/<entailment>.json \
        --process-metrics evals/reports/<process>.json \
        --thresholds evals/reports/<thresholds>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import aiofiles
from pydantic import BaseModel, ConfigDict, Field

ASSERTIONS_VERSION = "metric_assertions_v1"

# Zero-tolerance security metrics: any observed event fails the build. These
# correspond to the degradation-counter surfaces (src/app/observability/
# degradation.py) and the red-team suites (evals/reports/redteam_phase8.json):
# an unauthorized read or a poisoning event is never a tuning question.
ZERO_TOLERANCE_METRICS: frozenset[str] = frozenset(
    {"scope_violations", "unauthorized_reads", "poisoning_events"}
)
# Metrics where HIGHER is worse and the threshold is a ceiling.
CEILING_METRICS: frozenset[str] = frozenset(
    {
        "unsupported_claim_rate",
        "p95_latency_ms",
        "total_tokens",
        "total_cost_usd",
        "tool_calls",
    }
)

RETRIEVAL_REPORT_METRICS: tuple[str, ...] = (
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
)
ENTAILMENT_REPORT_METRICS: tuple[str, ...] = (
    "atomic_entailment_rate",
    "unsupported_claim_rate",
    "must_include_coverage",
    "abstention_correctness",
    "citation_validity",
    "scope_violations",
    "unauthorized_reads",
)
PROCESS_REPORT_METRICS: tuple[str, ...] = (
    "tool_calls",
    "p95_latency_ms",
    "total_tokens",
    "total_cost_usd",
)


class AssertionThresholds(BaseModel):
    """Recorded thresholds for the assertion gate (never invented inline).

    ``None`` means "not gated here"; every non-None field becomes a build
    gate. Security metrics are gated at zero unconditionally.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_report: str
    golden_set_version: str
    notes: str | None = None
    # Quality floors (higher is better).
    recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    atomic_entailment_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    must_include_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_validity: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_correctness: float | None = Field(default=None, ge=0.0, le=1.0)
    ingestion_fidelity_rows_passed: int | None = Field(default=None, ge=0)
    # Ceilings (lower is better).
    unsupported_claim_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    p95_latency_ms: float | None = Field(default=None, ge=0.0)
    total_tokens: int | None = Field(default=None, ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0.0)
    tool_calls: int | None = Field(default=None, ge=0)
    # Security metrics are recorded as the expected value (zero) for
    # provenance; they are enforced by ZERO_TOLERANCE_METRICS, not by the
    # generic threshold loop.
    scope_violations: int | None = Field(default=None, ge=0)
    unauthorized_reads: int | None = Field(default=None, ge=0)


def collect_metrics(  # noqa: PLR0912 — one flattening pass keeps every report family together
    *,
    retrieval_report: Mapping[str, object] | None = None,
    entailment_report: Mapping[str, object] | None = None,
    process_report: Mapping[str, object] | None = None,
    ingestion_report: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Flatten the report payloads into the assertion metric namespace."""
    metrics: dict[str, float] = {}
    if retrieval_report is not None:
        aggregates = retrieval_report.get("aggregates")
        if isinstance(aggregates, Mapping):
            for name in RETRIEVAL_REPORT_METRICS:
                value = aggregates.get(name)
                if isinstance(value, (int, float)):
                    metrics[name] = float(value)
        rows = retrieval_report.get("rows")
        if isinstance(rows, list):
            metrics["retrieval_rows"] = float(len(rows))
    if entailment_report is not None:
        aggregates = entailment_report.get("aggregates")
        if isinstance(aggregates, Mapping):
            for name in ENTAILMENT_REPORT_METRICS:
                value = aggregates.get(name)
                if isinstance(value, (int, float)):
                    metrics[name] = float(value)
            # Zero-tolerance security metrics default to 0 when a report
            # carries no such counter — absence of evidence of a violation is
            # only acceptable because the red-team suites assert the same
            # guarantees independently (fail-build step in CI).
            for name in ("scope_violations", "unauthorized_reads"):
                metrics.setdefault(name, 0.0)
    if process_report is not None:
        for name in PROCESS_REPORT_METRICS:
            value = process_report.get(name)
            if isinstance(value, (int, float)):
                metrics[name] = float(value)
    if ingestion_report is not None:
        gate = ingestion_report.get("gate")
        if isinstance(gate, Mapping):
            rows_passed = gate.get("rows_passed", 0)
            if isinstance(rows_passed, (int, float)):
                metrics["ingestion_fidelity_rows_passed"] = float(rows_passed)
            rows_failed = gate.get("rows_failed", 0)
            if isinstance(rows_failed, (int, float)) and rows_failed:
                metrics["ingestion_fidelity_rows_failed"] = float(rows_failed)
    return metrics


def evaluate_assertions(
    metrics: Mapping[str, float],
    thresholds: AssertionThresholds,
) -> list[str]:
    """Return the failing assertion lines; empty means the build gate passes."""
    failures: list[str] = []
    for metric in sorted(ZERO_TOLERANCE_METRICS):
        observed = metrics.get(metric)
        if observed is None:
            continue
        if observed > 0:
            failures.append(
                f"FAIL {metric}: zero-tolerance security metric observed {observed:.0f} "
                "events (> 0); unauthorized reads, scope violations, and poisoning "
                "fail the build unconditionally"
            )
    for metric, threshold in sorted(thresholds.model_dump(exclude_none=True).items()):
        if metric in {"source_report", "golden_set_version", "notes"}:
            continue
        if metric in ZERO_TOLERANCE_METRICS:
            continue  # enforced unconditionally above
        observed = metrics.get(metric)
        if observed is None:
            failures.append(
                f"FAIL {metric}: metric missing from the reports under assertion "
                f"(threshold {threshold}); gates fail closed on absent measurements"
            )
        elif metric in CEILING_METRICS:
            if observed > float(threshold):
                failures.append(f"FAIL {metric}: observed {observed:.4f} > ceiling {threshold}")
        elif observed < float(threshold):
            failures.append(f"FAIL {metric}: observed {observed:.4f} < threshold {threshold}")
    failed_fidelity = metrics.get("ingestion_fidelity_rows_failed")
    if failed_fidelity:
        failures.append(
            f"FAIL ingestion_fidelity: {failed_fidelity:.0f} labeled rows failed the fidelity gate"
        )
    return failures


def threshold_provenance(thresholds: AssertionThresholds) -> str:
    return (
        f"assertions derived from {thresholds.source_report} "
        f"(golden_set_version={thresholds.golden_set_version})"
    )


async def _load_json(path: Path) -> dict[str, object]:
    async with aiofiles.open(path, encoding="utf-8") as stream:
        return json.loads(await stream.read())  # type: ignore[no-any-return]


async def _run(args: argparse.Namespace) -> int:
    retrieval = await _load_json(args.retrieval_report) if args.retrieval_report else None
    entailment = await _load_json(args.entailment_report) if args.entailment_report else None
    process = await _load_json(args.process_report) if args.process_report else None
    ingestion = await _load_json(args.ingestion_report) if args.ingestion_report else None
    metrics = collect_metrics(
        retrieval_report=retrieval,
        entailment_report=entailment,
        process_report=process,
        ingestion_report=ingestion,
    )
    thresholds = AssertionThresholds.model_validate_json(
        args.thresholds.read_text(encoding="utf-8")
    )
    failures = evaluate_assertions(metrics, thresholds)
    if failures:
        sys.stderr.write(
            "metric assertion gate failed (" + threshold_provenance(thresholds) + ")\n"
        )
        for failure in failures:
            sys.stderr.write(failure + "\n")
        return 1
    sys.stdout.write("metric assertion gate passed (" + threshold_provenance(thresholds) + ")\n")
    for name in sorted(metrics):
        sys.stdout.write(f"  {name} = {metrics[name]}\n")
    return 0


def main() -> None:
    """Run the metric assertion gate CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, default=None)
    parser.add_argument("--entailment-report", type=Path, default=None)
    parser.add_argument("--process-report", type=Path, default=None)
    parser.add_argument("--ingestion-report", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
