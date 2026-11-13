"""Entailment evaluation and the reproducible entailment report schema (task 8.2).

The retrieval report (``report.py``) measures what the retriever surfaces; this
module measures what an ANSWER may claim: every atomic expected claim is judged
against its declared evidence chunks, every ``expected_answer_must_include``
token is checked against the evidence text, and abstention rows are checked for
schema-level consistency. The default judge is the deterministic offline
token-anchoring judge — reproducible with no model provider and no database —
so CI stays offline. The judge field on the report is part of the schema: a
live provider-backed NLI judge records its model id, prompt id, and prompt
hash on the same report shape when the quota-blocked live run resumes (see
``scripts/live_measurement/RUNBOOK.md``).

Usage:

    uv run python -m app.ai.evaluation.entailment evals/golden/answer_golden_v1.jsonl \
        --output evals/reports/<name>.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
from anyio import Path as AsyncPath
from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure

from .deterministic import _TOKEN_STOPWORDS, OfflineCorpusIndex, _tokenize
from .harness import resolve_commit
from .schema import NO_EVIDENCE_CASE_TYPES, load_validated_golden_set

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .schema import GoldenQuery

ENTAILMENT_REPORT_VERSION = "entailment_report_v1"
DETERMINISTIC_JUDGE_ID = "deterministic_token_anchor_v1"

# An atomic claim is "entailed" by the deterministic judge only when every
# content token of the claim is anchored in its declared evidence text;
# "partial" and "unsupported" are graded below those floors.
_ENTAILED_RATIO_FLOOR = 1.0
_PARTIAL_RATIO_FLOOR = 0.5


class EntailmentJudge(BaseModel):
    """The judge that produced a report's claim judgements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judge_id: str
    provider: str
    prompt_id: str
    prompt_sha256: str | None = None


DETERMINISTIC_JUDGE = EntailmentJudge(
    judge_id=DETERMINISTIC_JUDGE_ID,
    provider="offline",
    prompt_id="none (mechanical token anchoring, no prompt)",
)


class ClaimJudgement(BaseModel):
    """One atomic claim judged against its declared evidence chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    expected_support: str
    judged_support: str
    anchoring_ratio: float = Field(ge=0.0, le=1.0)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    missing_evidence_chunk_ids: list[str] = Field(default_factory=list)
    unanchored_tokens: list[str] = Field(default_factory=list)


class EntailmentQueryResult(BaseModel):
    """Per-query entailment outcome for one golden row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str | None
    query: str
    case_type: str
    abstention_expected: bool
    schema_consistent_abstention: bool | None = None
    must_include: list[str] = Field(default_factory=list)
    must_include_hits: list[str] = Field(default_factory=list)
    must_include_misses: list[str] = Field(default_factory=list)
    claims: list[ClaimJudgement] = Field(default_factory=list)
    citations_valid: bool = True
    citation_failures: list[str] = Field(default_factory=list)
    scope_violations: int = 0
    unauthorized_reads: int = 0


class EntailmentAggregates(BaseModel):
    """Aggregate entailment metrics over the scored rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    atomic_entailment_rate: float
    unsupported_claim_rate: float
    must_include_coverage: float
    # Live-answer-only: the offline deterministic path has no answerer, so
    # correctness of the abstain decision itself is recorded as None here and
    # scored by the assertion suite on live reports.
    abstention_correctness: float | None = None
    abstention_rows: int = 0
    citation_validity: float
    scope_violations: int = 0
    unauthorized_reads: int = 0
    claims_judged: int = 0


class EntailmentEvaluation(BaseModel):
    """Full deterministic entailment evaluation over a golden set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[EntailmentQueryResult]
    aggregates: EntailmentAggregates


class EntailmentReport(BaseModel):
    """A durable, reproducible description of one entailment evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_version: str = ENTAILMENT_REPORT_VERSION
    commit_identifier: str
    timestamp: datetime
    golden_set_version: str
    judge: EntailmentJudge
    rows: list[EntailmentQueryResult]
    aggregates: EntailmentAggregates

    @classmethod
    def from_evaluation(
        cls,
        evaluation: EntailmentEvaluation,
        *,
        commit_identifier: str,
        golden_set_version: str,
        judge: EntailmentJudge = DETERMINISTIC_JUDGE,
        timestamp: datetime | None = None,
    ) -> EntailmentReport:
        return cls(
            commit_identifier=commit_identifier,
            timestamp=timestamp or datetime.now(tz=UTC),
            golden_set_version=golden_set_version,
            judge=judge,
            rows=evaluation.rows,
            aggregates=evaluation.aggregates,
        )


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokenize(text) if token not in _TOKEN_STOPWORDS]


def judge_claim_deterministic(
    claim_id: str,
    claim_text: str,
    expected_support: str,
    evidence_texts: dict[str, str],
) -> ClaimJudgement:
    """Anchor one atomic claim in its declared evidence text.

    The deterministic judge never invents support: a token missing from every
    declared evidence chunk is recorded, the anchoring ratio is the fraction of
    content tokens anchored, and the graded support follows the floors. This is
    an anchoring PROXY for entailment — the report says so via the judge id.
    """
    tokens = _content_tokens(claim_text)
    haystack = " ".join(evidence_texts.values()).lower()
    unanchored = [token for token in tokens if token not in haystack]
    ratio = 1.0 if not tokens else 1.0 - len(unanchored) / len(tokens)
    if ratio >= _ENTAILED_RATIO_FLOOR:
        judged = "entailed"
    elif ratio >= _PARTIAL_RATIO_FLOOR:
        judged = "partial"
    else:
        judged = "unsupported"
    return ClaimJudgement(
        claim_id=claim_id,
        expected_support=expected_support,
        judged_support=judged,
        anchoring_ratio=round(ratio, 6),
        evidence_chunk_ids=sorted(evidence_texts),
        unanchored_tokens=unanchored,
    )


def evaluate_entailment(  # noqa: PLR0914 — one scoring pass keeps every claim/must-include/abstention signal together
    queries: Sequence[GoldenQuery],
    *,
    chunk_texts: dict[str, str],
) -> EntailmentEvaluation:
    """Score every golden row's answer expectations against its evidence text.

    ``chunk_texts`` maps chunk id to the canonical chunk text (offline: the
    recorded corpus index; live: the fetched canonical chunks).
    """
    rows: list[EntailmentQueryResult] = []
    claim_judgements: list[ClaimJudgement] = []
    must_include_total = 0
    must_include_hit_count = 0
    abstention_rows = 0
    citation_valid_rows = 0
    scope_violations = 0
    unauthorized_reads = 0
    for query in queries:
        claims = list(query.expected_claims)
        evidence_ids = sorted({cid for claim in claims for cid in claim.evidence_chunk_ids})
        missing_evidence = [cid for cid in evidence_ids if cid not in chunk_texts]
        evidence_texts = {cid: chunk_texts[cid] for cid in evidence_ids if cid in chunk_texts}
        judgements = [
            judge_claim_deterministic(
                claim.claim_id, claim.text, claim.expected_support, evidence_texts
            )
            for claim in claims
        ]
        claim_judgements.extend(judgements)
        if missing_evidence:
            judgements = [
                judgement.model_copy(
                    update={
                        "judged_support": "unsupported",
                        "missing_evidence_chunk_ids": missing_evidence,
                    }
                )
                for judgement in judgements
            ]
        citation_failures = [
            f"evidence chunk {cid} is not in the canonical chunk universe"
            for cid in missing_evidence
        ]
        hits: list[str] = []
        misses: list[str] = []
        evidence_haystack = " ".join(evidence_texts.values()).lower()
        for token in query.expected_answer_must_include:
            if token.lower() in evidence_haystack:
                hits.append(token)
            else:
                misses.append(token)
                citation_failures.append(
                    f"must-include token {token!r} is not anchored in the declared evidence"
                )
        abstention_expected = query.expected_abstention or query.case_type in NO_EVIDENCE_CASE_TYPES
        if abstention_expected:
            abstention_rows += 1
            # Schema-level consistency: an abstention row must not declare
            # claims and must expect a non-ok evidence status (V3). The live
            # abstain DECISION itself is scored only on live reports.
            consistent = not claims and query.expected_evidence_status is not None
            schema_consistent: bool | None = consistent
        else:
            schema_consistent = None
        if not citation_failures:
            citation_valid_rows += 1
        # Scope violations and unauthorized reads are observed events, not
        # declared expectations: the deterministic path observes none because
        # it never retrieves. The assertion suite (task 8.8) counts them from
        # retrieval reports (retrieved ∩ forbidden) and live traces.
        rows.append(
            EntailmentQueryResult(
                case_id=query.case_id,
                query=query.query,
                case_type=query.case_type,
                abstention_expected=abstention_expected,
                schema_consistent_abstention=schema_consistent,
                must_include=list(query.expected_answer_must_include),
                must_include_hits=hits,
                must_include_misses=misses,
                claims=judgements,
                citations_valid=not citation_failures,
                citation_failures=citation_failures,
            )
        )
        must_include_total += len(query.expected_answer_must_include)
        must_include_hit_count += len(hits)
    total_claims = len(claim_judgements)
    entailed = sum(1 for j in claim_judgements if j.judged_support == "entailed")
    unsupported = sum(1 for j in claim_judgements if j.judged_support == "unsupported")
    aggregates = EntailmentAggregates(
        atomic_entailment_rate=entailed / total_claims if total_claims else 0.0,
        unsupported_claim_rate=unsupported / total_claims if total_claims else 0.0,
        must_include_coverage=must_include_hit_count / must_include_total
        if must_include_total
        else 1.0,
        abstention_correctness=None,
        abstention_rows=abstention_rows,
        citation_validity=citation_valid_rows / len(rows) if rows else 0.0,
        scope_violations=scope_violations,
        unauthorized_reads=unauthorized_reads,
        claims_judged=total_claims,
    )
    return EntailmentEvaluation(rows=rows, aggregates=aggregates)


def prompt_fingerprint(prompt: str) -> str:
    """Stable sha256 of a live judge prompt, recorded on the report."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


async def write_entailment_report(report: EntailmentReport, path: Path) -> None:
    """Write a stable JSON entailment report."""
    await AsyncPath(path.parent).mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, mode="w", encoding="utf-8") as stream:
        await stream.write(report.model_dump_json(indent=2))
        await stream.write("\n")


async def run_entailment_eval(
    golden_path: Path,
    *,
    corpus_root: Path,
) -> tuple[EntailmentReport | None, str]:
    """Load, evaluate, and return an entailment report (or an error message)."""
    result = await load_validated_golden_set(golden_path)
    if isinstance(result, Failure):
        return None, result.failure().message
    golden_set = result.unwrap()
    index = OfflineCorpusIndex(corpus_root)
    evaluation = evaluate_entailment(golden_set.queries, chunk_texts=index.chunk_text_map())
    report = EntailmentReport.from_evaluation(
        evaluation,
        commit_identifier=resolve_commit(),
        golden_set_version=golden_set.version,
    )
    return report, ""


def main() -> None:
    """Run the deterministic entailment evaluation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden_set", type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evals/corpus/artifacts"),
        help="recorded corpus artifact root (pages.json per source)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report, error = asyncio.run(run_entailment_eval(args.golden_set, corpus_root=args.corpus))
    if report is None:
        sys.stderr.write(f"{error}\n")
        raise SystemExit(1)
    if args.output is not None:
        asyncio.run(write_entailment_report(report, args.output))
        sys.stdout.write(f"entailment report written: {args.output}\n")
    aggregates = report.aggregates
    sys.stdout.write(
        f"atomic_entailment_rate={aggregates.atomic_entailment_rate:.4f} "
        f"unsupported_claim_rate={aggregates.unsupported_claim_rate:.4f} "
        f"must_include_coverage={aggregates.must_include_coverage:.4f} "
        f"citation_validity={aggregates.citation_validity:.4f} "
        f"claims_judged={aggregates.claims_judged}\n"
    )


if __name__ == "__main__":
    main()
