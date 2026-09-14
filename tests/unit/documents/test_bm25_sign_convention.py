"""Pin the inverted BM25 sign convention (retrieval-sql task 2.1).

The keyword leg scores relevance as ``-1 * distance`` so that a *more* relevant
chunk carries a *more negative* raw score. The SQL therefore filters ``< 0`` and
orders ``ASC`` (most negative first). Inverting the ordering in the source must
make this test fail.
"""

from __future__ import annotations

import inspect
from operator import itemgetter
from pathlib import Path

from app.features.documents import repository as repository_module
from app.features.documents.fusion import RankedResultRow, reciprocal_rank_fusion

_REPO_SOURCE = Path(inspect.getfile(repository_module)).read_text(encoding="utf-8")


def test_bm25_relevance_expression_orders_ascending() -> None:
    assert "(-1 * (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')))" in _REPO_SOURCE
    assert (
        "ORDER BY (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')) ASC"
        in _REPO_SOURCE
    )


def test_bm25_relevance_expression_filters_below_zero() -> None:
    assert (
        "(c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')) < 0" in _REPO_SOURCE
    )


def test_more_negative_raw_score_outranks_less_negative() -> None:
    # Raw BM25 relevance scores: more negative means more relevant. The branch
    # ranks them ascending, so the -5.0 chunk is rank 1 and the -2.0 chunk rank 2.
    raw_scores = {"chunk_hot": -5.0, "chunk_warm": -2.0}
    ranked = [
        chunk_id
        for chunk_id, _ in sorted(raw_scores.items(), key=itemgetter(1))
    ]
    assert ranked == ["chunk_hot", "chunk_warm"]

    fused = reciprocal_rank_fusion(
        [
            RankedResultRow(chunk_id=chunk_id, score=raw, rank=rank)
            for rank, (chunk_id, raw) in enumerate(
                sorted(raw_scores.items(), key=itemgetter(1)), start=1
            )
        ],
        k=60,
        limit=2,
    )
    assert [item.chunk_id for item in fused] == ["chunk_hot", "chunk_warm"]
