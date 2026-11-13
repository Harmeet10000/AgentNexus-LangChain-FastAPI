"""Deterministic offline retrieval harness CLI (the CI quality gate runner).

Runs the golden corpus against the recorded corpus artifacts (no database, no
model provider) through :func:`run_retrieval_eval`, writes an
:class:`EvaluationReport`, and optionally enforces recorded thresholds so a
regression in measured recall/nDCG fails with the metric, observed value, and
threshold named.

Usage:

    uv run python -m app.ai.evaluation.harness evals/golden/legal_retrieval_v2.jsonl \
        --output evals/reports/<name>.json \
        --thresholds evals/reports/<name>.thresholds.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from returns.result import Failure

from .deterministic import OfflineCorpusIndex, fuse_branch_ranks
from .report import EvaluationReport, write_report
from .runner import BranchAwareRetrieval, BranchContribution, run_retrieval_eval
from .schema import load_validated_golden_set

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .schema import GoldenQuery


class ThresholdFile(BaseModel):
    """Recorded thresholds derived from a baseline report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_report: str
    golden_set_version: str
    cutoff: int
    thresholds: dict[str, float]
    notes: str | None = None

    @property
    def provenance_comment(self) -> str:
        return (
            f"thresholds derived from {self.source_report} "
            f"(golden_set_version={self.golden_set_version}, cutoff={self.cutoff})"
        )


def resolve_commit() -> str:
    """Return the current tree's commit id, or 'unknown' outside a repo.

    Reads ``.git`` directly (a directory in the normal layout, a pointer file
    in a worktree) rather than spawning a process, so the CLI stays free of
    subprocess usage.
    """
    git_path = Path(".git")
    if git_path.is_file():
        pointer = git_path.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir:"):
            return "unknown"
        git_path = Path(pointer.removeprefix("gitdir:").strip())
    head_path = git_path / "HEAD"
    if not head_path.is_file():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head or "unknown"
    ref_path = git_path / head.removeprefix("ref:").strip()
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip() or "unknown"
    packed_refs = git_path / "packed-refs"
    if packed_refs.is_file():
        wanted = head.removeprefix("ref:").strip()
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.strip().endswith(f" {wanted}"):
                return line.split()[0]
    return "unknown"


def parse_weights(raw: str | None) -> dict[str, float]:
    """Parse ``name=value`` comma pairs into a weight mapping."""
    if raw is None:
        return {}
    return {part.split("=", 1)[0].strip(): float(part.split("=", 1)[1]) for part in raw.split(",")}


async def _run(  # noqa: PLR0912 — one CLI pass keeps every gate check together
    args: argparse.Namespace,
    *,
    branch_names: Sequence[str],
    branch_weights: Sequence[float] | None,
    thresholds: ThresholdFile | None,
) -> int:
    golden_path = args.golden_set
    result = await load_validated_golden_set(golden_path)
    if isinstance(result, Failure):
        sys.stderr.write(f"{result.failure().message}\n")
        return 1
    golden_set = result.unwrap()
    index = OfflineCorpusIndex(args.corpus)
    if not index.chunk_ids and not args.allow_empty_corpus:
        # Fail closed BEFORE scoring: an empty chunk universe means the
        # harness is not wired to the corpus at all, and every query would
        # retrieve nothing — a wiring failure, never a zero baseline.
        sys.stderr.write(
            "wiring failure: the corpus root "
            f"{args.corpus} exposes no chunks; refusing to score an empty universe "
            "(pass --allow-empty-corpus to override deliberately)\n"
        )
        return 3
    cutoff = args.k
    if thresholds is not None:
        if thresholds.golden_set_version != golden_set.version:
            sys.stderr.write(
                f"threshold set mismatch: {thresholds.golden_set_version!r} recorded vs "
                f"{golden_set.version!r} loaded ({golden_path}); refusing to gate this run\n"
            )
            return 2
        if thresholds.cutoff != cutoff:
            sys.stderr.write(
                f"threshold cutoff mismatch: {thresholds.cutoff} recorded vs "
                f"{cutoff} for this run (--k {cutoff}); refusing to gate this run\n"
            )
            return 2

    async def retrieve(query: GoldenQuery) -> BranchAwareRetrieval:
        branch_ranks = index.branch_ranks(query.query, cutoff, branches=tuple(branch_names))
        return BranchAwareRetrieval(
            fused_chunk_ids=fuse_branch_ranks(branch_ranks, cutoff, weights=branch_weights),
            branches=[
                BranchContribution(branch=name, ranked_chunk_ids=ranks)
                for name, ranks in branch_ranks.items()
            ],
        )

    evaluation = await run_retrieval_eval(
        queries=golden_set.queries,
        retrieve=retrieve,
        cutoff=cutoff,
        branch_weights=branch_weights,
    )
    empty_rows = [row for row in evaluation.rows if not row.retrieved_chunk_ids]
    if len(empty_rows) == len(evaluation.rows) and evaluation.rows and not args.allow_empty_corpus:
        # Wiring-failure detection (task 8.3): when EVERY query retrieves zero
        # chunks the harness is not wired to the ranking path — a silently
        # all-zero report must never be written as a baseline.
        sys.stderr.write(
            "wiring failure: every query retrieved zero chunks; the retrieval path is "
            "not wired to the chunk universe. No report written.\n"
        )
        return 3
    report = EvaluationReport.from_evaluation(
        evaluation,
        commit_identifier=args.commit or resolve_commit(),
        golden_set_version=golden_set.version,
    )
    if args.output is not None:
        await write_report(report, args.output)
        sys.stdout.write(f"report written: {args.output}\n")

    if thresholds is not None:
        failures: list[str] = []
        for metric, threshold in sorted(thresholds.thresholds.items()):
            observed = getattr(report.aggregates, metric, None)
            if observed is None:
                failures.append(
                    f"FAIL {metric}: metric unavailable on the report aggregates "
                    f"(threshold {threshold})"
                )
            elif observed < threshold:
                failures.append(f"FAIL {metric}: observed {observed:.4f} < threshold {threshold}")
            else:
                sys.stdout.write(
                    f"PASS {metric}: observed {observed:.4f} >= threshold {threshold}\n"
                )
        if failures:
            sys.stderr.write(
                "retrieval quality gate failed (" + thresholds.provenance_comment + ")\n"
            )
            for failure in failures:
                sys.stderr.write(failure + "\n")
            return 1
    return 0


def main() -> None:
    """Run the offline deterministic retrieval harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden_set", type=Path)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("evals/corpus/artifacts"),
        help="recorded corpus artifact root (pages.json per source)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=10, help="evaluation cutoff")
    parser.add_argument(
        "--commit", type=str, default=None, help="commit id to record (default: the git HEAD id)"
    )
    parser.add_argument(
        "--branches",
        type=str,
        default="bm25,trigram",
        help="comma-separated branch legs to rank and fuse (bm25,trigram,graph)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="optional per-branch RRF weights as name=value pairs, e.g. bm25=1.0,graph=0.1",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="threshold JSON file; when given, the run gates on it and exits non-zero below it",
    )
    parser.add_argument(
        "--allow-empty-corpus",
        action="store_true",
        help="deliberately permit an empty chunk universe / all-empty retrieval "
        "(default: fail closed with a wiring-failure exit code 3)",
    )
    args = parser.parse_args()
    branch_names = [name.strip() for name in args.branches.split(",")]
    weight_map = parse_weights(args.weights)
    branch_weights = [weight_map.get(name, 1.0) for name in branch_names] or None
    thresholds: ThresholdFile | None = None
    if args.thresholds is not None:
        thresholds = ThresholdFile.model_validate_json(args.thresholds.read_text(encoding="utf-8"))
    raise SystemExit(
        asyncio.run(
            _run(
                args,
                branch_names=branch_names,
                branch_weights=branch_weights,
                thresholds=thresholds,
            )
        )
    )


if __name__ == "__main__":
    main()
