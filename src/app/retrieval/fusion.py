"""Hybrid search fusion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    *result_sets: list[RankedResultRow],
    k: int,
    limit: int,
    weights: Sequence[float] | None = None,
) -> list[RankedChunk]:
    """Fuse ranked result lists with the standard reciprocal-rank formula.

    ``weights`` is keyword-friendly and defaults to unweighted: when absent, every
    leg contributes ``1.0 / (k + rank)`` exactly as before, so existing callers are
    untouched. Pass one weight per leg to favour a leg's rankings.
    """
    leg_weights = _resolve_weights(count=len(result_sets), weights=weights)
    scores: dict[str, float] = {}

    for result_set, weight in zip(result_sets, leg_weights, strict=True):
        for result in result_set:
            chunk_id = result.chunk_id
            rank = result.rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        RankedChunk(chunk_id=chunk_id, score=score, rank=index)
        for index, (chunk_id, score) in enumerate(ordered[:limit], start=1)
    ]


def _resolve_weights(*, count: int, weights: Sequence[float] | None) -> list[float]:
    """Return one weight per leg, defaulting to unweighted."""
    if weights is None:
        return [1.0] * count
    resolved = list(weights)
    if len(resolved) != count:
        message = f"Expected one weight per leg ({count}), got {len(resolved)}"
        raise ValueError(message)
    return resolved
