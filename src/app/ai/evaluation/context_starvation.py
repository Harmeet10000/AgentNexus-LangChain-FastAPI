"""Context-starvation attribution over the retrieval baseline (G5 task 4.3).

The section-hydration adoption gate demands the measurement baseline attribute
recall loss to context starvation. This module computes that attribution with
the deterministic offline harness: for every golden query that missed an
expected chunk at the cutoff, it re-ranks each branch over the *full* chunk
universe (no cutoff) and classifies the miss:

- ``below_cut`` — the expected chunk appears in an extended branch ranking
  within the horizon but below the fused cutoff: the text signal exists and
  ranking sufficiency is the failure (the class of loss section hydration
  addresses by widening generation context — and that context starvation may
  drive through reranking/grading);
- ``no_signal`` — no branch's extended ranking surfaces the expected chunk
  within the horizon: the query's terms simply do not reach the chunk text
  (a recall loss hydration cannot fix — the chunk text needs the context, not
  the generator);
- ``not_in_corpus`` — the expected chunk id is not part of the recorded chunk
  universe at all.

Per-document-class counts are reported. Recorded report:
``evals/reports/context_starvation_legal_retrieval_v2_wave4.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from returns.result import Failure

from .deterministic import OfflineCorpusIndex
from .schema import load_validated_golden_set

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .schema import GoldenQuery

__all__ = [
    "ClassStarvation",
    "ContextStarvationReport",
    "attribute_context_starvation",
    "main",
]

#: Extended-ranking horizon: a miss whose expected chunk ranks within this
#: horizon in some branch is "retrievable but ranked below cut".
_HORIZON = 100


class ClassStarvation(BaseModel):
    """One document class's miss attribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_kind: str
    queries: int
    queries_with_misses: int
    below_cut_misses: int
    no_signal_misses: int
    not_in_corpus_misses: int


class ContextStarvationReport(BaseModel):
    """The recorded context-starvation attribution for the hydration gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    golden_set_version: str
    row_count: int
    cutoff: int
    horizon: int
    overall: ClassStarvation
    per_class: list[ClassStarvation]
    interpretation: str


def attribute_context_starvation(
    *,
    index: OfflineCorpusIndex,
    queries: Sequence[GoldenQuery],
    cutoff: int,
    horizon: int = _HORIZON,
) -> ContextStarvationReport:
    """Classify every missed expected chunk per query and document class."""

    def _classify(kind: str, row_indexes: list[int]) -> ClassStarvation:
        below = no_signal = not_in_corpus = 0
        queries_with_misses = 0
        universe = set(index.chunk_ids)
        for row_index in row_indexes:
            query = queries[row_index]
            expected = list(query.expected_chunk_ids)
            if not expected:
                continue
            misses = [
                chunk_id
                for chunk_id in expected
                if chunk_id not in index.fused_rank(query.query, cutoff)
            ]
            if not misses:
                continue
            queries_with_misses += 1
            for chunk_id in misses:
                if chunk_id not in universe:
                    not_in_corpus += 1
                    continue
                extended: set[str] = set()
                for branch in ("bm25", "trigram", "graph"):
                    extended.update(
                        index.branch_ranks(query.query, horizon, branches=(branch,))[branch]
                    )
                if chunk_id in extended:
                    below += 1
                else:
                    no_signal += 1
        return ClassStarvation(
            document_kind=kind,
            queries=len(row_indexes),
            queries_with_misses=queries_with_misses,
            below_cut_misses=below,
            no_signal_misses=no_signal,
            not_in_corpus_misses=not_in_corpus,
        )

    by_kind: dict[str, list[int]] = {}
    for row_index, query in enumerate(queries):
        by_kind.setdefault(str(query.document_kind), []).append(row_index)
    overall = _classify("all", list(range(len(queries))))
    per_class: list[ClassStarvation] = []
    for kind, indexes in sorted(by_kind.items()):
        per_class.append(_classify(kind, indexes))
    starvation_dominant = overall.below_cut_misses > (
        overall.no_signal_misses + overall.not_in_corpus_misses
    )
    return ContextStarvationReport(
        golden_set_version="recorded-run",
        row_count=len(queries),
        cutoff=cutoff,
        horizon=horizon,
        overall=overall,
        per_class=per_class,
        interpretation=(
            "Below-cut misses dominate the missed-chunk population: the text signal reaches "
            "most missed chunks but ranks them below the cutoff — the loss class section "
            "hydration addresses. This is CONSISTENT WITH context starvation, not proof of it: "
            "the offline harness measures ranking, not generation, so it cannot demonstrate the "
            "hydration effect itself. The hydration implementation lands DEFAULT-OFF; its "
            "measured recall/precision effect per document class remains pending one live run."
            if starvation_dominant
            else "Below-cut misses do not dominate the missed-chunk population: most missed "
            "chunks carry no branch signal at all. Hydration stays default-off; adoption "
            "evidence is pending a measurement that attributes recall loss to absent "
            "parent-section context."
        ),
    )


async def run_starvation_attribution(
    *,
    golden_set_path: Path,
    corpus_root: Path,
    cutoff: int = 10,
    horizon: int = _HORIZON,
) -> ContextStarvationReport | None:
    result = await load_validated_golden_set(golden_set_path)
    if isinstance(result, Failure):
        return None
    golden_set = result.unwrap()
    index = OfflineCorpusIndex(corpus_root)
    report = attribute_context_starvation(
        index=index, queries=golden_set.queries, cutoff=cutoff, horizon=horizon
    )
    return report.model_copy(update={"golden_set_version": golden_set.version})


async def _run(args: argparse.Namespace) -> int:
    report = await run_starvation_attribution(
        golden_set_path=args.golden_set,
        corpus_root=args.corpus,
        cutoff=args.k,
        horizon=args.horizon,
    )
    if report is None:
        sys.stderr.write("golden set failed validation\n")
        return 2
    output = args.output
    payload = report.model_dump_json(indent=2)
    if output is None:
        sys.stdout.write(payload + "\n")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(f"report written: {output}\n")
    return 0


def main() -> None:
    """Record the context-starvation attribution for the hydration gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden_set", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("evals/corpus/artifacts"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=_HORIZON)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
