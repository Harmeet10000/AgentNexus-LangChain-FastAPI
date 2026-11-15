"""Offline tests for the entailment evaluation and report schema (task 8.2).

The default judge is the deterministic token-anchoring judge: no model
provider, no database, byte-reproducible. The report schema carries the judge
identity (model id / prompt id / prompt hash) so the live NLI judge can adopt
the same report shape when the quota-blocked live run resumes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.ai.evaluation.entailment import (
    DETERMINISTIC_JUDGE,
    EntailmentAggregates,
    EntailmentReport,
    evaluate_entailment,
    judge_claim_deterministic,
    prompt_fingerprint,
    run_entailment_eval,
    write_entailment_report,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_ROOT = REPO_ROOT / "evals/corpus/artifacts"
ANSWER_GOLDEN = REPO_ROOT / "evals/golden/answer_golden_v1.jsonl"


def test_fully_anchored_claim_is_entailed() -> None:
    judgement = judge_claim_deterministic(
        "c1",
        "the Concessionaire shall furnish the Performance Security within 180 days",
        "entailed",
        {
            "SRC-TST:p1:c01": "provide to the Authority no later than 180 days an irrevocable "
            "guarantee; the Concessionaire shall furnish the Performance Security"
        },
    )
    assert judgement.judged_support == "entailed"
    assert judgement.anchoring_ratio == pytest.approx(1.0)
    assert judgement.unanchored_tokens == []


def test_paraphrased_claim_scores_partial_not_entailed() -> None:
    judgement = judge_claim_deterministic(
        "c1",
        "the Concessionaire shall provide the Performance Security 180 days",
        "entailed",
        {"SRC-TST:p1:c01": "the Concessionaire shall furnish the Performance Security"},
    )
    assert judgement.judged_support == "partial"
    assert judgement.anchoring_ratio == pytest.approx(4 / 7)
    assert "provide" in judgement.unanchored_tokens


def test_unanchored_claim_is_unsupported() -> None:
    judgement = judge_claim_deterministic(
        "c1",
        "the liquidated damages rate is 0.05 percent weekly",
        "entailed",
        {"c": "wholly unrelated text"},
    )
    assert judgement.judged_support == "unsupported"
    assert judgement.unanchored_tokens


def test_missing_evidence_chunk_is_recorded_not_invented() -> None:
    judgement = judge_claim_deterministic("c1", "payment within thirty days", "entailed", {})
    assert judgement.judged_support == "unsupported"
    assert judgement.anchoring_ratio == pytest.approx(0.0)


def test_number_anchoring_is_whole_token_not_substring() -> None:
    """A claimed ``30 days`` is not anchored by evidence saying ``130 days``:
    whole-token matching keeps number boundaries exactly where legal claims
    are graded. The same rule stops a word fragment anchoring a longer token."""
    substring_number = judge_claim_deterministic(
        "c1",
        "payment is due within 30 days",
        "entailed",
        {"c": "the payment is due within 130 days of the invoice"},
    )
    assert "30" in substring_number.unanchored_tokens
    assert substring_number.judged_support != "entailed"

    exact_number = judge_claim_deterministic(
        "c1",
        "payment is due within 30 days",
        "entailed",
        {"c": "the payment is due within 30 days of the invoice"},
    )
    assert exact_number.judged_support == "entailed"

    word_fragment = judge_claim_deterministic(
        "c1",
        "the termin clause applies",
        "entailed",
        {"c": "the termination clause applies on insolvency"},
    )
    assert "termin" in word_fragment.unanchored_tokens


def test_claims_are_judged_only_against_their_own_evidence() -> None:
    """A claim's sibling evidence chunk cannot supply its anchoring words: each
    claim is anchored only in the chunks it declares."""
    from app.ai.evaluation.schema import GoldenQuery

    row = GoldenQuery.model_validate(
        _golden_row(
            "what governs the security and damages",
            claims=[
                {
                    "claim_id": "sibling-anchored",
                    "text": "the security period is five years",
                    "expected_support": "entailed",
                    # Declares only the damages chunk — the security wording
                    # lives in the sibling chunk it does NOT declare.
                    "evidence_chunk_ids": ["SRC-TST:p1:c02"],
                    "gradability": "deterministic",
                },
                {
                    "claim_id": "own-evidence",
                    "text": "liquidated damages apply at 0.05 percent",
                    "expected_support": "entailed",
                    "evidence_chunk_ids": ["SRC-TST:p1:c02"],
                    "gradability": "deterministic",
                },
            ],
        )
    )
    evaluation = evaluate_entailment(
        [row],
        chunk_texts={
            "SRC-TST:p1:c01": "the security period is five years",
            "SRC-TST:p1:c02": "liquidated damages apply at 0.05 percent",
        },
    )

    by_id = {claim.claim_id: claim for claim in evaluation.rows[0].claims}
    assert by_id["sibling-anchored"].judged_support != "entailed"
    assert "security" in by_id["sibling-anchored"].unanchored_tokens
    assert by_id["own-evidence"].judged_support == "entailed"


def test_missing_evidence_replaces_the_grade_before_aggregation() -> None:
    """The aggregates never aggregate a stale grade: a claim whose declared
    evidence is missing is unsupported in ``claim_judgements`` — not the
    anchored grade its sibling chunks would have produced."""
    from app.ai.evaluation.schema import GoldenQuery

    row = GoldenQuery.model_validate(
        _golden_row(
            "what is the damages rate",
            claims=[
                {
                    "claim_id": "present",
                    "text": "liquidated damages apply at 0.05 percent",
                    "expected_support": "entailed",
                    "evidence_chunk_ids": ["SRC-TST:p1:c01"],
                    "gradability": "deterministic",
                },
                {
                    "claim_id": "missing-evidence",
                    # Identical wording to the present chunk: the pre-fix path
                    # anchored this claim against the query-level evidence pool
                    # and extended the stale entailed grade into the aggregates.
                    "text": "liquidated damages apply at 0.05 percent",
                    "expected_support": "entailed",
                    "evidence_chunk_ids": ["SRC-TST:missing:c01"],
                    "gradability": "deterministic",
                },
            ],
        )
    )
    evaluation = evaluate_entailment(
        [row],
        chunk_texts={"SRC-TST:p1:c01": "liquidated damages apply at 0.05 percent"},
    )

    claims = {claim.claim_id: claim for claim in evaluation.rows[0].claims}
    assert claims["missing-evidence"].judged_support == "unsupported"
    assert claims["missing-evidence"].missing_evidence_chunk_ids == ["SRC-TST:missing:c01"]
    assert evaluation.aggregates.atomic_entailment_rate == pytest.approx(0.5)
    assert evaluation.aggregates.unsupported_claim_rate == pytest.approx(0.5)
    assert evaluation.aggregates.claims_judged == 2


def _golden_row(
    query: str, *, claims: list[dict[str, object]], **extra: object
) -> dict[str, object]:
    return {
        "query": query,
        "expected_chunk_ids": ["SRC-TST:p1:c01"],
        "expected_document_ids": ["SRC-TST"],
        "document_kind": "contracts",
        "difficulty": "easy",
        "notes": "entailment probe",
        "awaiting_sme_expansion": True,
        "case_type": "direct_answer",
        "expected_claims": claims,
        **extra,
    }


def test_evaluate_entailment_aggregates_rows_and_abstention_consistency() -> None:
    from app.ai.evaluation.schema import GoldenQuery

    good = _golden_row(
        "what is the security period",
        claims=[
            {
                "claim_id": "c1",
                "text": "the performance security shall cover defects for five years",
                "expected_support": "entailed",
                "evidence_chunk_ids": ["SRC-TST:p1:c01"],
                "gradability": "deterministic",
            }
        ],
    )
    abstention = _golden_row(
        "what is the moon made of",
        claims=[],
        case_type="unanswerable",
        expected_chunk_ids=[],
        expected_document_ids=[],
        retrieval_relevance="not_applicable",
        claim_entailment="not_applicable",
        expected_abstention=True,
        expected_evidence_status="no_evidence",
    )
    queries = [
        GoldenQuery.model_validate(good),
        GoldenQuery.model_validate(abstention),
    ]
    evaluation = evaluate_entailment(
        queries,
        chunk_texts={
            "SRC-TST:p1:c01": "the performance security shall cover defects for five years"
        },
    )
    assert evaluation.aggregates.claims_judged == 1
    assert evaluation.aggregates.atomic_entailment_rate == pytest.approx(1.0)
    assert evaluation.aggregates.abstention_rows == 1
    assert evaluation.rows[1].schema_consistent_abstention is True
    # The offline deterministic path never fabricates a live abstain decision.
    assert evaluation.aggregates.abstention_correctness is None


def test_report_roundtrip_records_judge_commit_and_timestamp(tmp_path: Path) -> None:
    evaluation_rows: list = []
    aggregates = EntailmentAggregates(
        atomic_entailment_rate=1.0,
        unsupported_claim_rate=0.0,
        must_include_coverage=1.0,
        abstention_correctness=None,
        abstention_rows=0,
        citation_validity=1.0,
        scope_violations=0,
        unauthorized_reads=0,
        claims_judged=0,
    )
    from app.ai.evaluation.entailment import EntailmentEvaluation

    report = EntailmentReport.from_evaluation(
        EntailmentEvaluation(rows=evaluation_rows, aggregates=aggregates),
        commit_identifier="deadbeef",
        golden_set_version="answer_golden_v1",
        judge=DETERMINISTIC_JUDGE,
        timestamp=datetime(2026, 11, 14, tzinfo=UTC),
    )
    out = tmp_path / "report.json"
    import asyncio

    asyncio.run(write_entailment_report(report, out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["report_version"] == "entailment_report_v1"
    assert payload["commit_identifier"] == "deadbeef"
    assert payload["golden_set_version"] == "answer_golden_v1"
    assert payload["judge"]["judge_id"] == "deterministic_token_anchor_v1"
    assert payload["judge"]["prompt_sha256"] is None


def test_live_judge_prompt_fingerprint_is_stable() -> None:
    assert prompt_fingerprint("same prompt") == prompt_fingerprint("same prompt")
    assert prompt_fingerprint("a") != prompt_fingerprint("b")


@pytest.mark.asyncio
async def test_answer_golden_set_entailment_run_is_citation_valid_offline() -> None:
    """The full offline run over the answer gold set is reproducible and valid."""
    report, error = await run_entailment_eval(ANSWER_GOLDEN, corpus_root=CORPUS_ROOT)
    assert error == ""
    assert report is not None
    assert report.golden_set_version == "answer_golden_v1"
    assert report.aggregates.citation_validity == pytest.approx(1.0)
    assert report.aggregates.must_include_coverage == pytest.approx(1.0)
    assert report.aggregates.claims_judged > 0
    assert all(row.case_id for row in report.rows)
    assert report.judge == DETERMINISTIC_JUDGE
