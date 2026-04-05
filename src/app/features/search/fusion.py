"""Hybrid search fusion helpers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RankedResultRow(BaseModel):
    """Minimal ranked row returned by a search branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    score: float
    rank: int


class RankedChunk(BaseModel):
    """Fused rank metadata for a search chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    score: float
    rank: int


def reciprocal_rank_fusion(
    bm25_results: list[RankedResultRow],
    vector_results: list[RankedResultRow],
    *,
    k: int,
    limit: int,
) -> list[RankedChunk]:
    """Fuse ranked result lists with the standard reciprocal-rank formula."""
    scores: dict[str, float] = {}

    for result_set in (bm25_results, vector_results):
        for result in result_set:
            chunk_id = result.chunk_id
            rank = result.rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        RankedChunk(chunk_id=chunk_id, score=score, rank=index)
        for index, (chunk_id, score) in enumerate(ordered[:limit], start=1)
    ]
