"""Deterministic, provider-independent experiment harness for retrieval evals.

The harness deliberately stops at an adapter boundary.  Production systems can
provide an adapter later, but this module never creates a model client, reads
confidential data, or silently turns an unvalidated golden set into evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from anyio import Path as AsyncPath
from pydantic import BaseModel, ConfigDict, Field, model_validator
from returns.result import Failure

from .metrics import ndcg_at_k, recall_at_k
from .schema import (
    NO_EVIDENCE_CASE_TYPES,
    GoldenItemProvenance,
    load_validated_golden_set,
    validate_golden_set,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .schema import GoldenQuery, GoldenSet

type VariantKind = Literal[
    "inline_ask",
    "evidence_pipeline",
    "deterministic_retrieve",
    "compiled_retrieval_graph",
    "deterministic_preamble",
    "contextualized_preamble",
]
type EvidencePolicy = Literal["graph_indexed_only", "mechanically_verified"]
type SupportStatus = Literal["entailed", "partial", "contradicted", "unsupported"]
type ThresholdOperator = Literal["gte", "gt", "lte", "lt", "eq"]
type ThresholdAction = Literal["fail", "kill"]


class JudgeSpec(BaseModel):
    """Reproducibility metadata for a generator or entailment judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ThresholdRule(BaseModel):
    """A predeclared pass/fail or kill rule over an aggregate metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(min_length=1)
    operator: ThresholdOperator
    value: float
    action: ThresholdAction
    variant: str | None = None
    stratum: str | None = None


class VariantConfig(BaseModel):
    """One system/policy configuration under comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    kind: VariantKind
    evidence_policy: EvidencePolicy
    comparison_group: str = Field(default="default", min_length=1)


class ExperimentManifest(BaseModel):
    """Immutable configuration for one reproducible experiment run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    golden_set_path: str = Field(min_length=1)
    golden_set_version: str = Field(min_length=1)
    smoke_only: bool = False
    fixture: Literal["synthetic_seed", "adapter"] = "adapter"
    cutoff: int = Field(default=10, gt=0)
    variants: list[VariantConfig] = Field(min_length=1)
    generator_judge: JudgeSpec
    entailment_judge: JudgeSpec
    thresholds: list[ThresholdRule] = Field(default_factory=list)
    adapter_path: str | None = None

    @model_validator(mode="after")
    def validate_smoke_fixture(self) -> ExperimentManifest:
        if self.fixture == "synthetic_seed" and not self.smoke_only:
            message = "synthetic_seed fixture is permitted only for smoke_only manifests"
            raise ValueError(message)
        return self


class EvidenceRef(BaseModel):
    """A source reference returned by an adapter, without source text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    page_no: int | None = None
    document_version: str | None = None
    pipeline_version: str | None = None


class ClaimObservation(BaseModel):
    """A generated atomic claim and its independently assessed support."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    status: SupportStatus
    evidence_ids: list[str] = Field(default_factory=list)


class CitationObservation(BaseModel):
    """Citation validation outcome produced by an adapter or deterministic checker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    valid: bool


class Coverage(BaseModel):
    """Document/index coverage captured for a single query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    documents_requested: int = Field(default=0, ge=0)
    documents_ready: int = Field(default=0, ge=0)
    documents_failed: int = Field(default=0, ge=0)
    graph_indexed_documents: int = Field(default=0, ge=0)

    @property
    def ratio(self) -> float:
        """Return ready/requested coverage; zero means no coverage was claimed."""
        if self.documents_requested == 0:
            return 0.0
        return min(self.documents_ready / self.documents_requested, 1.0)


class RuntimeStats(BaseModel):
    """Provider-independent runtime telemetry for one query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_calls: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AdapterOutput(BaseModel):
    """Provider-neutral output required from an experiment adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    claims: list[ClaimObservation] = Field(default_factory=list)
    citations: list[CitationObservation] = Field(default_factory=list)
    abstained: bool = False
    coverage: Coverage = Field(default_factory=Coverage)
    runtime: RuntimeStats = Field(default_factory=RuntimeStats)
    evidence_policy: EvidencePolicy


class ExperimentAdapter(Protocol):
    """Minimum adapter contract; implementations may wrap fakes or real systems."""

    async def run(self, query: GoldenQuery, variant: VariantConfig) -> AdapterOutput: ...


class CaseMetrics(BaseModel):
    """Metrics captured for one query/variant pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_recall_at_k: float
    retrieval_ndcg_at_k: float
    claim_entailment_accuracy: float | None
    claim_support_rate: float
    unsupported_claim_rate: float
    abstained: float
    correct_abstention: float
    citation_validity: float
    coverage: float
    tool_calls: float
    model_calls: float
    latency_ms: float
    tokens: float
    cost_usd: float


class CaseObservation(BaseModel):
    """Raw and scored output for one query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    stratum: str
    variant: str
    evidence_policy: EvidencePolicy
    retrieved_chunk_ids: list[str]
    evidence: list[EvidenceRef] = Field(default_factory=list)
    golden_provenance: GoldenItemProvenance | None = None
    claims: list[ClaimObservation]
    citations: list[CitationObservation]
    abstained: bool
    coverage: Coverage
    runtime: RuntimeStats
    metrics: CaseMetrics


class AggregateMetrics(BaseModel):
    """Macro averages plus count telemetry for a variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_count: int
    retrieval_recall_at_k: float
    retrieval_ndcg_at_k: float
    claim_entailment_accuracy: float | None
    claim_support_rate: float
    unsupported_claim_rate: float
    abstention_rate: float
    correct_abstention_rate: float
    citation_validity: float
    coverage: float
    tool_calls: float
    model_calls: float
    latency_ms: float
    tokens: float
    cost_usd: float


class VariantEvaluation(BaseModel):
    """All observations and aggregates for one variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: VariantConfig
    rows: list[CaseObservation]
    aggregates: AggregateMetrics
    by_stratum: dict[str, AggregateMetrics]


class ThresholdResult(BaseModel):
    """Evaluation of one configured decision rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: ThresholdRule
    actual: float | None
    passed: bool
    killed: bool
    message: str


class ExperimentVerdict(BaseModel):
    """Overall decision, preserving every failed or killed rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["passed", "failed", "killed"]
    checks: list[ThresholdResult]


class PolicyComparison(BaseModel):
    """Explicit comparison between two evidence policies/variants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_variant: str
    candidate_variant: str
    baseline_policy: EvidencePolicy
    candidate_policy: EvidencePolicy
    metric_deltas: dict[str, float | None]


class ExperimentReport(BaseModel):
    """Durable report containing provenance, judges, observations, and verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_name: str
    manifest_version: str
    golden_set_version: str
    smoke_only: bool
    decisional: bool
    provenance_schema_version: str
    timestamp: datetime
    generator_judge: JudgeSpec
    entailment_judge: JudgeSpec
    variants: list[VariantEvaluation]
    verdict: ExperimentVerdict
    comparisons: list[VariantComparison] = Field(default_factory=list)


class VariantComparison(BaseModel):
    """Paired variant deltas, retained by query stratum."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_group: str
    baseline_variant: str
    candidate_variant: str
    by_stratum: dict[str, dict[str, float | None]]


def stratum_for(query: GoldenQuery) -> str:
    """Classify a query without inventing labels absent from the gold set."""
    if query.case_type in NO_EVIDENCE_CASE_TYPES:
        return "no_evidence"
    if query.case_type == "near_miss_clause":
        return "near_miss"
    notes = query.notes.lower()
    if "conflicting" in notes:
        return "conflicting_authority"
    if "multi-hop" in notes or "multi hop" in notes:
        return "multi_hop"
    if len(set(query.expected_document_ids)) > 1:
        return "multi_document"
    return query.difficulty


def stratify_queries(queries: Sequence[GoldenQuery]) -> dict[str, list[GoldenQuery]]:
    """Return stable, insertion-ordered strata for reporting and A/B analysis."""
    result: dict[str, list[GoldenQuery]] = defaultdict(list)
    for query in queries:
        result[stratum_for(query)].append(query)
    return dict(result)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return _mean(present) if present else None


def _aggregate(rows: Sequence[CaseObservation]) -> AggregateMetrics:
    return AggregateMetrics(
        query_count=len(rows),
        retrieval_recall_at_k=_mean([row.metrics.retrieval_recall_at_k for row in rows]),
        retrieval_ndcg_at_k=_mean([row.metrics.retrieval_ndcg_at_k for row in rows]),
        claim_entailment_accuracy=_mean_optional(
            [row.metrics.claim_entailment_accuracy for row in rows]
        ),
        claim_support_rate=_mean([row.metrics.claim_support_rate for row in rows]),
        unsupported_claim_rate=_mean([row.metrics.unsupported_claim_rate for row in rows]),
        abstention_rate=_mean([row.metrics.abstained for row in rows]),
        correct_abstention_rate=_mean([row.metrics.correct_abstention for row in rows]),
        citation_validity=_mean([row.metrics.citation_validity for row in rows]),
        coverage=_mean([row.metrics.coverage for row in rows]),
        tool_calls=_mean([row.metrics.tool_calls for row in rows]),
        model_calls=_mean([row.metrics.model_calls for row in rows]),
        latency_ms=_mean([row.metrics.latency_ms for row in rows]),
        tokens=_mean([row.metrics.tokens for row in rows]),
        cost_usd=_mean([row.metrics.cost_usd for row in rows]),
    )


def _case_metrics(
    query: GoldenQuery,
    output: AdapterOutput,
    *,
    cutoff: int,
) -> CaseMetrics:
    expected = set(query.expected_chunk_ids)
    claims = output.claims
    supported = sum(claim.status in {"entailed", "partial"} for claim in claims)
    unsupported = sum(claim.status in {"unsupported", "contradicted"} for claim in claims)
    entailment: float | None = None
    if query.claim_entailment != "not_applicable" and claims:
        entailment = sum(claim.status == query.claim_entailment for claim in claims) / len(claims)
    citations = output.citations
    evidence_ids = {item.evidence_id for item in output.evidence}
    no_evidence = query.case_type in NO_EVIDENCE_CASE_TYPES
    return CaseMetrics(
        retrieval_recall_at_k=recall_at_k(output.retrieved_chunk_ids, expected, cutoff),
        retrieval_ndcg_at_k=ndcg_at_k(output.retrieved_chunk_ids, expected, cutoff),
        claim_entailment_accuracy=entailment,
        claim_support_rate=supported / len(claims) if claims else 0.0,
        unsupported_claim_rate=unsupported / len(claims) if claims else 0.0,
        abstained=float(output.abstained),
        correct_abstention=float(output.abstained == no_evidence),
        citation_validity=sum(
            citation.valid and citation.evidence_id in evidence_ids for citation in citations
        )
        / len(citations)
        if citations
        else 0.0,
        coverage=output.coverage.ratio,
        tool_calls=float(output.runtime.tool_calls),
        model_calls=float(output.runtime.model_calls),
        latency_ms=output.runtime.latency_ms,
        tokens=float(output.runtime.tokens),
        cost_usd=output.runtime.cost_usd,
    )


def _metric_value(metrics: AggregateMetrics, name: str) -> float | None:
    value = getattr(metrics, name, None)
    return value if isinstance(value, (float, int)) else None


def evaluate_thresholds(
    rules: Sequence[ThresholdRule],
    evaluations: Sequence[VariantEvaluation],
) -> ExperimentVerdict:
    """Apply only rules declared before execution; missing metrics fail safely."""
    checks: list[ThresholdResult] = []
    for rule in rules:
        targets = [
            evaluation
            for evaluation in evaluations
            if rule.variant in {None, evaluation.variant.name}
        ]
        actual_values = [
            value
            for evaluation in targets
            for aggregate in [
                evaluation.by_stratum.get(rule.stratum)
                if rule.stratum is not None
                else evaluation.aggregates
            ]
            if aggregate is not None
            and (value := _metric_value(aggregate, rule.metric)) is not None
        ]
        actual = _mean(actual_values) if actual_values else None
        passed = False
        if actual is not None:
            passed = {
                "gte": actual >= rule.value,
                "gt": actual > rule.value,
                "lte": actual <= rule.value,
                "lt": actual < rule.value,
                "eq": actual == rule.value,
            }[rule.operator]
        killed = not passed and rule.action == "kill"
        message = (
            f"{rule.metric} {rule.operator} {rule.value}"
            f"{f' [{rule.stratum}]' if rule.stratum else ''}: actual={actual}"
            if actual is not None
            else f"{rule.metric}: metric unavailable"
        )
        checks.append(
            ThresholdResult(
                rule=rule,
                actual=actual,
                passed=passed,
                killed=killed,
                message=message,
            )
        )
    if any(check.killed for check in checks):
        status: Literal["passed", "failed", "killed"] = "killed"
    elif any(not check.passed for check in checks):
        status = "failed"
    else:
        status = "passed"
    return ExperimentVerdict(status=status, checks=checks)


def compare_policies(
    report: ExperimentReport,
    *,
    baseline_variant: str,
    candidate_variant: str,
) -> PolicyComparison:
    """Compare two variants while retaining their evidence-policy labels."""
    by_name = {evaluation.variant.name: evaluation for evaluation in report.variants}
    baseline = by_name[baseline_variant]
    candidate = by_name[candidate_variant]
    metric_names = (
        "retrieval_recall_at_k",
        "retrieval_ndcg_at_k",
        "claim_entailment_accuracy",
        "claim_support_rate",
        "unsupported_claim_rate",
        "abstention_rate",
        "correct_abstention_rate",
        "citation_validity",
        "coverage",
        "tool_calls",
        "model_calls",
        "latency_ms",
        "tokens",
        "cost_usd",
    )
    deltas: dict[str, float | None] = {}
    for name in metric_names:
        left = _metric_value(baseline.aggregates, name)
        right = _metric_value(candidate.aggregates, name)
        deltas[name] = right - left if left is not None and right is not None else None
    return PolicyComparison(
        baseline_variant=baseline_variant,
        candidate_variant=candidate_variant,
        baseline_policy=baseline.variant.evidence_policy,
        candidate_policy=candidate.variant.evidence_policy,
        metric_deltas=deltas,
    )


def _comparison_metrics(
    baseline: AggregateMetrics,
    candidate: AggregateMetrics,
) -> dict[str, float | None]:
    metric_names = (
        "retrieval_recall_at_k",
        "retrieval_ndcg_at_k",
        "claim_entailment_accuracy",
        "claim_support_rate",
        "unsupported_claim_rate",
        "abstention_rate",
        "correct_abstention_rate",
        "citation_validity",
        "coverage",
        "tool_calls",
        "model_calls",
        "latency_ms",
        "tokens",
        "cost_usd",
    )
    return {
        metric: (
            candidate_value - baseline_value
            if (baseline_value := _metric_value(baseline, metric)) is not None
            and (candidate_value := _metric_value(candidate, metric)) is not None
            else None
        )
        for metric in metric_names
    }


def _build_comparisons(evaluations: Sequence[VariantEvaluation]) -> list[VariantComparison]:
    grouped: dict[str, list[VariantEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[evaluation.variant.comparison_group].append(evaluation)
    comparisons: list[VariantComparison] = []
    for group, members in grouped.items():
        if len(members) < 2:
            continue
        baseline, *candidates = members
        for candidate in candidates:
            strata = sorted(set(baseline.by_stratum) | set(candidate.by_stratum))
            comparisons.append(
                VariantComparison(
                    comparison_group=group,
                    baseline_variant=baseline.variant.name,
                    candidate_variant=candidate.variant.name,
                    by_stratum={
                        stratum: _comparison_metrics(
                            baseline.by_stratum.get(stratum, _aggregate([])),
                            candidate.by_stratum.get(stratum, _aggregate([])),
                        )
                        for stratum in strata
                    },
                )
            )
    return comparisons


async def run_experiment(
    *,
    manifest: ExperimentManifest,
    golden_set: GoldenSet,
    adapter: ExperimentAdapter,
) -> ExperimentReport:
    """Run every configured variant against a provenance-validated gold set."""
    validate_golden_set(golden_set)
    if golden_set.version != manifest.golden_set_version:
        message = "manifest golden_set_version does not match the loaded golden set"
        raise ValueError(message)
    if manifest.fixture == "synthetic_seed" and any(
        query.provenance is None or query.provenance.source_type != "synthetic"
        for query in golden_set.queries
    ):
        message = "synthetic_seed experiments require synthetic-only golden rows"
        raise ValueError(message)
    evaluations: list[VariantEvaluation] = []
    for variant in manifest.variants:
        rows: list[CaseObservation] = []
        for query in golden_set.queries:
            provenance = query.provenance
            if provenance is None:
                message = f"query lacks validated provenance: {query.query}"
                raise ValueError(message)
            started = time.perf_counter()
            output = await adapter.run(query, variant)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if output.evidence_policy != variant.evidence_policy:
                message = (
                    f"adapter policy mismatch for {variant.name}: "
                    f"{output.evidence_policy} != {variant.evidence_policy}"
                )
                raise ValueError(message)
            runtime = output.runtime.model_copy(
                update={"latency_ms": max(output.runtime.latency_ms, elapsed_ms)}
            )
            output = output.model_copy(update={"runtime": runtime})
            rows.append(
                CaseObservation(
                    query=query.query,
                    stratum=stratum_for(query),
                    variant=variant.name,
                    evidence_policy=output.evidence_policy,
                    retrieved_chunk_ids=output.retrieved_chunk_ids,
                    evidence=output.evidence,
                    golden_provenance=provenance,
                    claims=output.claims,
                    citations=output.citations,
                    abstained=output.abstained,
                    coverage=output.coverage,
                    runtime=runtime,
                    metrics=_case_metrics(query, output, cutoff=manifest.cutoff),
                )
            )
        by_stratum = {
            stratum: _aggregate([row for row in rows if row.stratum == stratum])
            for stratum in sorted({row.stratum for row in rows})
        }
        evaluations.append(
            VariantEvaluation(
                variant=variant,
                rows=rows,
                aggregates=_aggregate(rows),
                by_stratum=by_stratum,
            )
        )
    return ExperimentReport(
        manifest_name=manifest.name,
        manifest_version=manifest.manifest_version,
        golden_set_version=golden_set.version,
        smoke_only=manifest.smoke_only,
        decisional=not manifest.smoke_only,
        provenance_schema_version=golden_set.provenance_schema_version or "unknown",
        timestamp=datetime.now(tz=UTC),
        generator_judge=manifest.generator_judge,
        entailment_judge=manifest.entailment_judge,
        variants=evaluations,
        verdict=evaluate_thresholds(manifest.thresholds, evaluations),
        comparisons=_build_comparisons(evaluations),
    )


class SyntheticSeedAdapter:
    """A deterministic smoke adapter; it is intentionally not a quality benchmark."""

    @staticmethod
    async def run(query: GoldenQuery, variant: VariantConfig) -> AdapterOutput:
        expected = list(query.expected_chunk_ids)
        documents = list(query.expected_document_ids)
        no_evidence = query.case_type in NO_EVIDENCE_CASE_TYPES
        retrieved = [] if no_evidence else expected
        evidence = [
            EvidenceRef(
                evidence_id=f"smoke:{chunk_id}",
                chunk_id=chunk_id,
                document_id=documents[index % len(documents)] if documents else "smoke-document",
            )
            for index, chunk_id in enumerate(retrieved)
        ]
        return AdapterOutput(
            retrieved_chunk_ids=retrieved,
            evidence=evidence,
            citations=[
                CitationObservation(evidence_id=item.evidence_id, valid=True) for item in evidence
            ],
            abstained=no_evidence,
            coverage=Coverage(
                documents_requested=len(documents),
                documents_ready=len(documents),
                graph_indexed_documents=len(documents)
                if variant.evidence_policy == "graph_indexed_only"
                else 0,
            ),
            runtime=RuntimeStats(tool_calls=1, model_calls=0),
            evidence_policy=variant.evidence_policy,
        )


async def write_experiment_report(report: ExperimentReport, path: Path) -> None:
    """Write a stable JSON report without requiring provider dependencies."""
    destination = AsyncPath(path)
    await destination.parent.mkdir(parents=True, exist_ok=True)
    await destination.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _load_adapter(path: str) -> ExperimentAdapter:
    """Load a caller-owned adapter factory from ``module:attribute``."""
    module_name, separator, attribute_name = path.partition(":")
    if not separator or not module_name or not attribute_name:
        message = "adapter path must use module:attribute syntax"
        raise ValueError(message)
    factory = getattr(importlib.import_module(module_name), attribute_name)
    adapter = factory() if callable(factory) else factory
    if not hasattr(adapter, "run"):
        message = "adapter must expose an async run(query, variant) method"
        raise TypeError(message)
    return cast("ExperimentAdapter", adapter)


async def _run_cli(
    manifest_path: Path,
    output_path: Path | None,
    adapter_path: str | None = None,
) -> int:
    manifest_text = await AsyncPath(manifest_path).read_text(encoding="utf-8")
    manifest = ExperimentManifest.model_validate_json(manifest_text)
    gold_path = (manifest_path.parent / manifest.golden_set_path).resolve()
    result = await load_validated_golden_set(gold_path)
    if isinstance(result, Failure):
        sys.stderr.write(f"{result.failure().message}\n")
        return 1
    adapter: ExperimentAdapter
    if manifest.fixture == "synthetic_seed":
        adapter = SyntheticSeedAdapter()
    else:
        configured_adapter = adapter_path or manifest.adapter_path
        if configured_adapter is None:
            sys.stderr.write("adapter fixture required for non-smoke manifests\n")
            return 2
        try:
            adapter = _load_adapter(configured_adapter)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            sys.stderr.write(f"unable to load adapter: {exc}\n")
            return 2
    report = await run_experiment(manifest=manifest, golden_set=result.unwrap(), adapter=adapter)
    destination = output_path or Path("evals/reports") / f"{manifest.name}.json"
    await write_experiment_report(report, destination)
    sys.stdout.write(
        json.dumps({"status": report.verdict.status, "report": str(destination)}) + "\n"
    )
    return 0 if report.verdict.status == "passed" else 1


def main() -> None:
    """Run a manifest-backed deterministic experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="module:attribute adapter factory for non-smoke manifests",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run_cli(args.manifest, args.output, args.adapter)))
