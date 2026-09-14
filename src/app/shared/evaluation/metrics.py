"""Pure, binary-relevance retrieval metrics."""

from collections.abc import Sequence, Set
from math import log2


def _top_k(ranked_ids: Sequence[str], k: int) -> Sequence[str]:
    """Return the requested prefix, treating non-positive cutoffs as empty."""
    return ranked_ids[: max(k, 0)]


def recall_at_k(ranked_ids: Sequence[str], expected_ids: Set[str], k: int) -> float:
    """Return the fraction of expected identifiers present in the first ``k`` ranks."""
    if not expected_ids:
        return 0.0
    hits = len(set(_top_k(ranked_ids, k)) & expected_ids)
    return hits / len(expected_ids)


def precision_at_k(ranked_ids: Sequence[str], expected_ids: Set[str], k: int) -> float:
    """Return relevant identifiers per requested rank position in the first ``k`` ranks."""
    if k <= 0:
        return 0.0
    hits = len(set(_top_k(ranked_ids, k)) & expected_ids)
    return hits / k


def reciprocal_rank(ranked_ids: Sequence[str], expected_ids: Set[str], k: int) -> float:
    """Return the reciprocal rank of the first expected identifier within ``k``."""
    for rank, identifier in enumerate(_top_k(ranked_ids, k), start=1):
        if identifier in expected_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], expected_ids: Set[str], k: int) -> float:
    """Return binary normalized discounted cumulative gain at ``k``."""
    if k <= 0 or not expected_ids:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, identifier in enumerate(_top_k(ranked_ids, k), start=1):
        if identifier in expected_ids and identifier not in seen:
            dcg += 1.0 / log2(rank + 1)
            seen.add(identifier)
    ideal_hits = min(k, len(expected_ids))
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg
