from __future__ import annotations

from pathlib import Path

import pytest
from anyio import Path as AsyncPath
from pydantic import ValidationError

from app.ai.evaluation.experiment import (
    AdapterOutput,
    ClaimObservation,
    Coverage,
    ExperimentManifest,
    RuntimeStats,
    SyntheticSeedAdapter,
    ThresholdRule,
    VariantConfig,
    compare_policies,
    evaluate_thresholds,
    run_experiment,
    stratify_queries,
)
from app.ai.evaluation.schema import (
    GoldenItemProvenance,
    GoldenQuery,
    GoldenSet,
    PolicyOverride,
    load_golden_set,
)


def _provenance() -> GoldenItemProvenance:
    return GoldenItemProvenance(
        source_type="synthetic",
        source_reference="eval-fixture://experiment-test",
        permission_status="synthetic_owned",
        license_terms_status="not_required",
        is_public=False,
        is_synthetic=True,
        human_review_status="engineer_reviewed",
        annotators={
            "engineer_relevance": "engineer",
            "legal_sme_entailment": "pending-sme",
        },
    )


def _query(
    text: str,
    *,
    chunks: list[str] | None = None,
    documents: list[str] | None = None,
    notes: str = "fixture",
    case_type: str = "standard",
) -> GoldenQuery:
    return GoldenQuery(
        query=text,
        expected_chunk_ids=chunks or [f"chunk-{text}"],
        expected_document_ids=documents or [f"document-{text}"],
        jurisdiction="India",
        document_kind="contracts",
        difficulty="easy",
        notes=notes,
        awaiting_sme_expansion=True,
        case_type=case_type,  # type: ignore[arg-type]
        claim_entailment="entailed",
        provenance=_provenance(),
    )


def _manifest(*variants: VariantConfig) -> ExperimentManifest:
    return ExperimentManifest(
        manifest_version="experiment_v1",
        name="test-experiment",
        golden_set_path="gold.jsonl",
        golden_set_version="gold_v1",
        smoke_only=True,
        fixture="synthetic_seed",
        variants=list(variants),
        generator_judge={
            "name": "generator",
            "model": "fake",
            "prompt_version": "p1",
            "prompt": "deterministic",
            "version": "fake-v1",
        },
        entailment_judge={
            "name": "entailment",
            "model": "fake",
            "prompt_version": "p1",
            "prompt": "deterministic",
            "version": "fake-v1",
        },
        thresholds=[],
    )


class FakeAdapter:
    def __init__(self, *, wrong_for: str | None = None) -> None:
        self.wrong_for = wrong_for

    async def run(self, query: GoldenQuery, variant: VariantConfig) -> AdapterOutput:
        chunk_ids = [] if query.query == self.wrong_for else query.expected_chunk_ids
        return AdapterOutput(
            retrieved_chunk_ids=chunk_ids,
            claims=[ClaimObservation(claim_id="claim-1", status="entailed")],
            citations=[],
            coverage=Coverage(documents_requested=1, documents_ready=1),
            runtime=RuntimeStats(tool_calls=2, model_calls=1, input_tokens=4, output_tokens=2),
            evidence_policy=variant.evidence_policy,
        )


class AbstainAdapter:
    """Abstains for every query, or only for one designated query."""

    def __init__(self, *, abstain_for: str | None = None) -> None:
        self.abstain_for = abstain_for

    async def run(self, query: GoldenQuery, variant: VariantConfig) -> AdapterOutput:
        abstained = self.abstain_for is None or query.query == self.abstain_for
        return AdapterOutput(
            retrieved_chunk_ids=[],
            claims=[],
            citations=[],
            abstained=abstained,
            coverage=Coverage(documents_requested=1, documents_ready=1),
            runtime=RuntimeStats(tool_calls=1, model_calls=0),
            evidence_policy=variant.evidence_policy,
        )


def _abstention_query(text: str, **overrides: object) -> GoldenQuery:
    """A direct-answer v2 row whose query itself declares expected abstention."""
    row: dict[str, object] = {
        "query": text,
        "expected_chunk_ids": [f"chunk-{text}"],
        "expected_document_ids": [f"document-{text}"],
        "jurisdiction": "India",
        "document_kind": "contracts",
        "difficulty": "easy",
        "notes": "fixture",
        "awaiting_sme_expansion": False,
        "case_type": "direct_answer",
        "retrieval_relevance": "relevant",
        "claim_entailment": "not_applicable",
        "expected_abstention": True,
        "expected_evidence_status": "no_evidence",
        "provenance": _provenance(),
    }
    row.update(overrides)
    return GoldenQuery(**row)  # type: ignore[arg-type]


def _no_evidence_query(text: str, **overrides: object) -> GoldenQuery:
    """An index_not_ready v2 row with a graph-indexed policy override."""
    row: dict[str, object] = {
        "query": text,
        "expected_chunk_ids": [],
        "expected_document_ids": [],
        "jurisdiction": "India",
        "document_kind": "contracts",
        "difficulty": "hard",
        "notes": "fixture",
        "awaiting_sme_expansion": False,
        "case_type": "index_not_ready",
        "retrieval_relevance": "not_applicable",
        "claim_entailment": "not_applicable",
        "expected_abstention": True,
        "expected_evidence_status": "index_not_ready",
        "expected_coverage_ratio_min": 0.0,
        "policy_overrides": {
            "graph_indexed_only": PolicyOverride(
                expected_abstention=False,
                expected_evidence_status="index_not_ready",
            )
        },
        "provenance": _provenance(),
    }
    row.update(overrides)
    return GoldenQuery(**row)  # type: ignore[arg-type]


def test_stratification_preserves_multi_document_and_special_cases() -> None:
    queries = [
        _query("easy"),
        _query("multi", documents=["one", "two"]),
        _query("hop", notes="multi-hop obligation chain"),
        _query("near", case_type="near_miss"),
    ]

    strata = stratify_queries(queries)

    assert list(strata) == ["easy", "multi_document", "multi_hop", "near_miss"]
    assert [query.query for query in strata["multi_document"]] == ["multi"]


@pytest.mark.asyncio
async def test_run_experiment_aggregates_metrics_and_runtime() -> None:
    variant = VariantConfig(
        name="pipeline",
        kind="evidence_pipeline",
        evidence_policy="mechanically_verified",
    )
    golden_set = GoldenSet(
        version="gold_v1",
        provenance_schema_version="golden_provenance_v1",
        queries=[_query("one"), _query("two")],
    )

    report = await run_experiment(
        manifest=_manifest(variant),
        golden_set=golden_set,
        adapter=FakeAdapter(wrong_for="two"),
    )

    metrics = report.variants[0].aggregates
    assert metrics.query_count == 2
    assert metrics.retrieval_recall_at_k == pytest.approx(0.5)
    assert metrics.tool_calls == pytest.approx(2.0)
    assert metrics.tokens == pytest.approx(6.0)
    assert metrics.claim_entailment_accuracy == pytest.approx(1.0)
    assert report.variants[0].by_stratum["easy"].query_count == 2


@pytest.mark.asyncio
async def test_policy_comparison_preserves_evidence_policy() -> None:
    graph = VariantConfig(
        name="graph",
        kind="inline_ask",
        evidence_policy="graph_indexed_only",
    )
    verified = VariantConfig(
        name="verified",
        kind="evidence_pipeline",
        evidence_policy="mechanically_verified",
    )
    golden_set = GoldenSet(
        version="gold_v1",
        provenance_schema_version="golden_provenance_v1",
        queries=[_query("one")],
    )
    report = await run_experiment(
        manifest=_manifest(graph, verified), golden_set=golden_set, adapter=FakeAdapter()
    )

    comparison = compare_policies(report, baseline_variant="graph", candidate_variant="verified")

    assert comparison.baseline_policy == "graph_indexed_only"
    assert comparison.candidate_policy == "mechanically_verified"
    assert comparison.metric_deltas["retrieval_recall_at_k"] == pytest.approx(0.0)


def test_threshold_verdict_supports_fail_and_kill() -> None:
    variant = VariantConfig(
        name="pipeline",
        kind="evidence_pipeline",
        evidence_policy="mechanically_verified",
    )
    aggregate = {
        "query_count": 1,
        "retrieval_recall_at_k": 0.5,
        "retrieval_ndcg_at_k": 0.5,
        "claim_entailment_accuracy": None,
        "claim_support_rate": 0.5,
        "unsupported_claim_rate": 0.5,
        "abstention_rate": 0.0,
        "correct_abstention_rate": 0.0,
        "citation_validity": 1.0,
        "coverage": 1.0,
        "tool_calls": 1.0,
        "model_calls": 1.0,
        "latency_ms": 1.0,
        "tokens": 1.0,
        "cost_usd": 0.0,
    }
    from app.ai.evaluation.experiment import AggregateMetrics, VariantEvaluation

    evaluation = VariantEvaluation(
        variant=variant,
        rows=[],
        aggregates=AggregateMetrics(**aggregate),
        by_stratum={},
    )
    verdict = evaluate_thresholds(
        [
            ThresholdRule(
                metric="retrieval_recall_at_k",
                operator="gte",
                value=0.8,
                action="fail",
            ),
            ThresholdRule(
                metric="unsupported_claim_rate",
                operator="lte",
                value=0.1,
                action="kill",
            ),
        ],
        [evaluation],
    )

    assert verdict.status == "killed"
    assert [check.passed for check in verdict.checks] == [False, False]


def test_threshold_can_target_a_query_stratum() -> None:
    variant = VariantConfig(
        name="pipeline",
        kind="evidence_pipeline",
        evidence_policy="mechanically_verified",
    )
    aggregate = {
        "query_count": 1,
        "retrieval_recall_at_k": 1.0,
        "retrieval_ndcg_at_k": 1.0,
        "claim_entailment_accuracy": None,
        "claim_support_rate": 1.0,
        "unsupported_claim_rate": 0.0,
        "abstention_rate": 0.0,
        "correct_abstention_rate": 0.0,
        "citation_validity": 1.0,
        "coverage": 1.0,
        "tool_calls": 1.0,
        "model_calls": 1.0,
        "latency_ms": 1.0,
        "tokens": 1.0,
        "cost_usd": 0.0,
    }
    from app.ai.evaluation.experiment import AggregateMetrics, VariantEvaluation

    evaluation = VariantEvaluation(
        variant=variant,
        rows=[],
        aggregates=AggregateMetrics(**aggregate),
        by_stratum={"near_miss": AggregateMetrics(**{**aggregate, "retrieval_recall_at_k": 0.0})},
    )
    verdict = evaluate_thresholds(
        [
            ThresholdRule(
                metric="retrieval_recall_at_k",
                operator="gte",
                value=0.5,
                action="kill",
                stratum="near_miss",
            )
        ],
        [evaluation],
    )

    assert verdict.status == "killed"
    assert verdict.checks[0].actual == pytest.approx(0.0)


def test_manifest_rejects_non_smoke_synthetic_fixture() -> None:
    with pytest.raises(ValidationError):
        _manifest().model_copy(update={"smoke_only": False})


@pytest.mark.asyncio
async def test_run_experiment_rejects_unvalidated_provenance() -> None:
    variant = VariantConfig(
        name="pipeline",
        kind="evidence_pipeline",
        evidence_policy="mechanically_verified",
    )
    golden_set = GoldenSet(version="gold_v1", queries=[_query("one")])

    with pytest.raises(ValueError, match="provenance_schema_version"):
        await run_experiment(
            manifest=_manifest(variant), golden_set=golden_set, adapter=FakeAdapter()
        )


def test_smoke_manifest_references_only_seed_fixture() -> None:
    path = Path("evals/experiments/legal_retrieval_smoke.json")
    manifest = ExperimentManifest.model_validate_json(path.read_text(encoding="utf-8"))

    assert manifest.smoke_only is True
    assert manifest.fixture == "synthetic_seed"
    assert "legal_retrieval_v1" in manifest.golden_set_version
    assert {variant.comparison_group for variant in manifest.variants} == {
        "answering-path",
        "retrieval-path",
        "preamble",
    }


@pytest.mark.asyncio
async def test_smoke_report_is_explicitly_non_decisional() -> None:
    path = Path("evals/experiments/legal_retrieval_smoke.json")
    manifest = ExperimentManifest.model_validate_json(
        await AsyncPath(path).read_text(encoding="utf-8")
    )
    result = await load_golden_set(Path("evals/golden/legal_retrieval_v1.jsonl"))
    golden_set = result.unwrap()

    report = await run_experiment(
        manifest=manifest,
        golden_set=golden_set,
        adapter=SyntheticSeedAdapter(),
    )

    assert report.smoke_only is True
    assert report.decisional is False
    assert report.provenance_schema_version == "golden_provenance_v1"
    assert report.variants[0].rows[0].golden_provenance.source_type == "synthetic"
    assert {comparison.comparison_group for comparison in report.comparisons} == {
        "answering-path",
        "retrieval-path",
        "preamble",
    }


@pytest.mark.asyncio
async def test_v2_row_honors_query_expected_abstention() -> None:
    """correct_abstention is scored against the query field, not the case type."""
    variant = VariantConfig(
        name="pipeline",
        kind="evidence_pipeline",
        evidence_policy="mechanically_verified",
    )
    golden_set = GoldenSet(
        version="gold_v1",
        provenance_schema_version="golden_provenance_v1",
        queries=[_abstention_query("expect-abstention")],
    )

    # adapter never abstains: the query's expected_abstention=true makes this
    # incorrect (the old case-type lookup would have scored it correct)
    report = await run_experiment(
        manifest=_manifest(variant), golden_set=golden_set, adapter=FakeAdapter()
    )
    assert report.variants[0].rows[0].metrics.correct_abstention == pytest.approx(0.0)

    report = await run_experiment(
        manifest=_manifest(variant),
        golden_set=golden_set,
        adapter=AbstainAdapter(abstain_for="expect-abstention"),
    )
    assert report.variants[0].rows[0].metrics.correct_abstention == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_policy_override_abstention_applies_under_matching_variant() -> None:
    """A policy_overrides entry keyed by the variant's policy wins over the base."""
    graph = VariantConfig(name="graph", kind="inline_ask", evidence_policy="graph_indexed_only")
    verified = VariantConfig(
        name="verified",
        kind="evidence_pipeline",
        evidence_policy="mechanically_verified",
    )
    golden_set = GoldenSet(
        version="gold_v1",
        provenance_schema_version="golden_provenance_v1",
        queries=[_no_evidence_query("index-not-ready")],
    )

    report = await run_experiment(
        manifest=_manifest(graph, verified),
        golden_set=golden_set,
        adapter=AbstainAdapter(abstain_for="index-not-ready"),
    )

    by_variant = {evaluation.variant.name: evaluation for evaluation in report.variants}
    # graph_indexed_only override expects NO abstention, so abstaining is wrong
    graph_row = by_variant["graph"].rows[0]
    assert graph_row.metrics.correct_abstention == pytest.approx(0.0)
    # mechanically_verified has no override: the base expected_abstention holds
    verified_row = by_variant["verified"].rows[0]
    assert verified_row.metrics.correct_abstention == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_v1_rows_keep_case_type_abstention_fallback() -> None:
    """Rows without expected_abstention keep the case-type lookup behaviour."""
    variant = VariantConfig(
        name="pipeline",
        kind="evidence_pipeline",
        evidence_policy="mechanically_verified",
    )
    # a v1-shaped row neither declares expected_abstention nor policy_overrides
    v1_row = GoldenQuery(
        query="legacy-missing",
        expected_chunk_ids=[],
        expected_document_ids=[],
        jurisdiction="India",
        document_kind="statutes",
        difficulty="hard",
        notes="legacy fixture",
        awaiting_sme_expansion=False,
        case_type="statute_not_in_corpus",
        retrieval_relevance="not_applicable",
        claim_entailment="not_applicable",
        provenance=_provenance(),
    )
    answerable = _query("legacy-answerable")
    assert "expected_abstention" not in v1_row.model_fields_set
    assert "expected_abstention" not in answerable.model_fields_set
    assert v1_row.policy_overrides == {}
    golden_set = GoldenSet(
        version="gold_v1",
        provenance_schema_version="golden_provenance_v1",
        queries=[v1_row, answerable],
    )

    report = await run_experiment(
        manifest=_manifest(variant),
        golden_set=golden_set,
        adapter=AbstainAdapter(),  # abstains for every query
    )

    by_query = {row.query: row.metrics for row in report.variants[0].rows}
    # no-evidence case type: abstaining is the correct behaviour
    assert by_query["legacy-missing"].correct_abstention == pytest.approx(1.0)
    # direct_answer case type: abstaining is not expected
    assert by_query["legacy-answerable"].correct_abstention == pytest.approx(0.0)
