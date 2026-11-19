"""Measured adoption comparison for contextual chunk retrieval (G2 task 2.2).

Runs the deterministic offline harness twice over the same golden set — once
over the recorded corpus artifacts, once with a recorded context overlay
prepended to each chunk's searchable text — and reports the per-document-class
delta. This measures the *retrieval effect* of the preamble integration (the
harness measures ranking, not generation): the overlay lines stand in for the
context sentences the ingest step would generate, so a recorded comparison here
is necessary but NOT sufficient for default-on. The generation-quality half of
the adoption decision needs one live run before the flag flips; the recorded
report and its decision field say so explicitly.

Usage:

    uv run python -m app.ai.evaluation.context_comparison \
        evals/golden/legal_retrieval_v2.jsonl \
        --context-overlay evals/corpus/context_overlay_fixture.json \
        --output evals/reports/contextual_preamble_legal_retrieval_v2_wave4.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
from pydantic import BaseModel, ConfigDict
from returns.result import Failure

from .deterministic import OfflineCorpusIndex, chunks_for_page, fuse_branch_ranks
from .runner import BranchAwareRetrieval, BranchContribution, run_retrieval_eval
from .schema import load_validated_golden_set

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .runner import RetrievalEvaluation
    from .schema import GoldenQuery, GoldenSet

#: Branch set and cutoff mirror the recorded wave-4 baseline report so the
#: deltas are comparable to `evals/reports/baseline_legal_retrieval_v2_wave4.json`.
_COMPARISON_BRANCHES = ("bm25", "trigram")
_COMPARISON_CUTOFF = 10

# Fixture overlay bound: the context line carries the source's opening block
# (its title/parties region) — real document-level text, mechanically derived,
# so the fixture is deterministic and reproducible from the corpus artifacts.
_OVERLAY_SOURCE_CHARS = 200


class ClassComparison(BaseModel):
    """One document class's baseline-vs-context comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_kind: str
    queries: int
    baseline_recall_at_k: float
    with_context_recall_at_k: float
    delta_recall_at_k: float
    baseline_ndcg_at_k: float
    with_context_ndcg_at_k: float
    delta_ndcg_at_k: float
    baseline_reciprocal_rank: float
    with_context_reciprocal_rank: float
    delta_reciprocal_rank: float


class ContextComparisonReport(BaseModel):
    """The recorded adoption comparison for the contextual-preamble feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    golden_set_version: str
    row_count: int
    cutoff: int
    overlay_description: str
    overall: ClassComparison
    per_class: list[ClassComparison]
    decision: str
    # ``None`` on reports written before the CLI wired ``--commit`` through.
    commit: str | None = None


def build_fixture_overlay(corpus_root: Path) -> tuple[dict[str, str], str]:
    """Derive the recorded fixture context lines from the corpus artifacts.

    Every chunk of a source receives one shared document-level line built from
    that source's opening block — the mechanical analogue of the ingest
    preamble: real document-level context prepended to the searchable text,
    one derivation per document. Deterministic by construction; NOT a stand-in
    for generated sentence quality.
    """
    overlay: dict[str, str] = {}
    for pages_path in sorted(corpus_root.glob("*/pages.json")):
        source_id = pages_path.parent.name
        payload = json.loads(pages_path.read_text(encoding="utf-8"))
        pages = payload.get("pages", [])
        opening = (
            " ".join(pages[0].get("text", "").split())[:_OVERLAY_SOURCE_CHARS] if pages else ""
        )
        if not opening:
            continue
        context_line = (
            f"Document context: this chunk is from {source_id}, whose document opens: {opening}."
        )
        for page in pages:
            for ordinal in range(1, len(chunks_for_page(page.get("text", ""))) + 1):
                overlay[f"{source_id}:p{page['page_no']}:c{ordinal:02d}"] = context_line
    description = (
        "fixture overlay: one shared document-level line per source (opening block of the "
        "recorded artifact), prepended to every chunk's searchable text"
    )
    return overlay, description


def _class_row(
    kind: str,
    indexes: Sequence[int],
    baseline: RetrievalEvaluation,
    with_context: RetrievalEvaluation,
) -> ClassComparison:
    base_rows = [baseline.rows[index] for index in indexes]
    ctx_rows = [with_context.rows[index] for index in indexes]

    def _mean(rows: Sequence[object], metric: str) -> float:
        values = [getattr(row.metrics, metric) for row in rows]  # ty: ignore[unresolved-attribute]
        return sum(values) / len(values) if values else 0.0

    return ClassComparison(
        document_kind=kind,
        queries=len(indexes),
        baseline_recall_at_k=_mean(base_rows, "recall_at_k"),
        with_context_recall_at_k=_mean(ctx_rows, "recall_at_k"),
        delta_recall_at_k=_mean(ctx_rows, "recall_at_k") - _mean(base_rows, "recall_at_k"),
        baseline_ndcg_at_k=_mean(base_rows, "ndcg_at_k"),
        with_context_ndcg_at_k=_mean(ctx_rows, "ndcg_at_k"),
        delta_ndcg_at_k=_mean(ctx_rows, "ndcg_at_k") - _mean(base_rows, "ndcg_at_k"),
        baseline_reciprocal_rank=_mean(base_rows, "reciprocal_rank"),
        with_context_reciprocal_rank=_mean(ctx_rows, "reciprocal_rank"),
        delta_reciprocal_rank=_mean(ctx_rows, "reciprocal_rank")
        - _mean(base_rows, "reciprocal_rank"),
    )


async def _evaluate(
    index: OfflineCorpusIndex, queries: Sequence[GoldenQuery], cutoff: int
) -> RetrievalEvaluation:
    async def retrieve(query: GoldenQuery) -> BranchAwareRetrieval:
        ranks = index.branch_ranks(query.query, cutoff, branches=_COMPARISON_BRANCHES)
        return BranchAwareRetrieval(
            fused_chunk_ids=fuse_branch_ranks(ranks, cutoff),
            branches=[
                BranchContribution(branch=name, ranked_chunk_ids=ids) for name, ids in ranks.items()
            ],
        )

    return await run_retrieval_eval(queries=queries, retrieve=retrieve, cutoff=cutoff)


async def compare_context_overlay(
    *,
    golden_set: GoldenSet,
    corpus_root: Path,
    overlay: dict[str, str],
    overlay_description: str,
    cutoff: int = _COMPARISON_CUTOFF,
    commit: str | None = None,
) -> ContextComparisonReport:
    """Run baseline and overlaid evaluations and report per-class deltas."""
    queries = golden_set.queries
    baseline = await _evaluate(OfflineCorpusIndex(corpus_root), queries, cutoff)
    with_context = await _evaluate(
        OfflineCorpusIndex(corpus_root, context_overlay=overlay), queries, cutoff
    )
    by_kind: dict[str, list[int]] = {}
    for index, query in enumerate(queries):
        by_kind.setdefault(str(query.document_kind), []).append(index)
    overall = _class_row("all", range(len(queries)), baseline, with_context)
    per_class = [
        _class_row(kind, indexes, baseline, with_context)
        for kind, indexes in sorted(by_kind.items())
    ]
    return ContextComparisonReport(
        golden_set_version=golden_set.version,
        row_count=len(queries),
        cutoff=cutoff,
        overlay_description=overlay_description,
        commit=commit,
        overall=overall,
        per_class=per_class,
        decision=(
            "PENDING LIVE RUN: the offline comparison measures the retrieval effect of recorded "
            "fixture context lines (ranking only, no generation quality). Default stays OFF "
            "(DOCUMENTS_CONTEXTUAL_PREAMBLE_ENABLED=False) until a live generation run completes "
            "the adoption evidence; a per-class regression on this comparison reverts any default."
        ),
    )


async def run_comparison(
    *,
    golden_set_path: Path,
    corpus_root: Path,
    overlay_path: Path,
    cutoff: int = _COMPARISON_CUTOFF,
    commit: str | None = None,
) -> ContextComparisonReport | None:
    """Load the golden set and a recorded overlay, and produce the comparison.

    Returns None when the golden set fails validation (the CLI reports it).
    ``cutoff`` and ``commit`` pass the CLI's ``--k``/``--commit`` options
    through: the requested cutoff drives the evaluation (instead of the
    default cutoff silently overruling ``--k``) and the commit is recorded on
    the report so the decision record names the code it measured.
    """
    result = await load_validated_golden_set(golden_set_path)
    if isinstance(result, Failure):
        return None
    golden_set = result.unwrap()
    async with aiofiles.open(overlay_path, encoding="utf-8") as stream:
        overlay = json.loads(await stream.read())
    if not isinstance(overlay, dict):
        message = "context overlay must be a JSON object mapping chunk ids to lines"
        raise TypeError(message)
    return await compare_context_overlay(
        golden_set=golden_set,
        corpus_root=corpus_root,
        overlay={str(key): str(value) for key, value in overlay.items()},
        overlay_description=f"recorded overlay {overlay_path}",
        cutoff=cutoff,
        commit=commit,
    )


async def _run(args: argparse.Namespace) -> int:
    comparison = await run_comparison(
        golden_set_path=args.golden_set,
        corpus_root=args.corpus,
        overlay_path=args.context_overlay,
        cutoff=args.k,
        commit=args.commit,
    )
    if comparison is None:
        sys.stderr.write("golden set failed validation\n")
        return 2
    output = args.output or Path(
        f"contextual_preamble_{comparison.golden_set_version}.comparison.json"
    )
    # The comparison file IS the decision record: per-class deltas plus the
    # adoption decision, written beside the recorded baseline reports.
    await _write_comparison(comparison, output)
    sys.stdout.write(f"comparison written: {output}\n")
    return 0


async def _write_comparison(report: ContextComparisonReport, path: Path) -> None:
    from aiofiles import open as aio_open  # noqa: PLC0415 — matches report.py's writer

    path.parent.mkdir(parents=True, exist_ok=True)
    async with aio_open(path, mode="w", encoding="utf-8") as stream:
        await stream.write(report.model_dump_json(indent=2))
        await stream.write("\n")


def main() -> None:
    """Run the baseline-vs-context comparison over the golden corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden_set", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("evals/corpus/artifacts"))
    parser.add_argument(
        "--context-overlay",
        type=Path,
        required=True,
        help="JSON object mapping chunk ids to recorded context lines",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=_COMPARISON_CUTOFF)
    parser.add_argument("--commit", type=str, default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
