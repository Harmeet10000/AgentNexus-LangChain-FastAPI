"""Pure, binary-relevance retrieval metrics and annotation-agreement math."""

from collections.abc import Mapping, Sequence, Set
from math import log2
from typing import Final


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


# Ordinal scale of the retrieval-relevance dimension, from strongest to weakest
# evidence claim.  Adjacency on this scale defines the coarse agreement
# patterns reported by the wave-3 IAA tooling.  Mirrors the
# ``RetrievalRelevance`` literal in ``app.ai.evaluation.schema`` (redeclared so
# this module stays import-free of the schema layer).
RELEVANCE_AGREEMENT_SCALE: Final[tuple[str, ...]] = (
    "relevant",
    "partially_relevant",
    "irrelevant",
    "not_applicable",
)


def kappa_from_margins(observed_agreement: float, expected_agreement: float) -> float:
    """Return Cohen's kappa from pre-computed observed/expected agreement.

    ``(p_o - p_e) / (1 - p_e)`` with the degenerate-case contract: when the
    expected agreement reaches 1.0 the denominator vanishes, so a totally
    agreeing sample is defined as kappa 1.0 and any non-total agreement on a
    single-category sample is defined as kappa 0.0.
    """
    if expected_agreement >= 1.0:
        return 1.0 if observed_agreement >= 1.0 else 0.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def cohen_kappa_unweighted(matrix: Mapping[str, Mapping[str, int]]) -> float:
    """Return unweighted Cohen's kappa for a confusion matrix.

    ``matrix`` maps rater-A labels to rater-B label counts (rows = rater A,
    columns = rater B).  Expected agreement is the sum over labels of the
    product of the two raters' marginal proportions; an empty matrix scores
    0.0.
    """
    total = sum(sum(row.values()) for row in matrix.values())
    if total == 0:
        return 0.0
    observed = sum(row.get(label, 0) for label, row in matrix.items()) / total
    row_totals = {label: sum(row.values()) for label, row in matrix.items()}
    col_totals: dict[str, int] = {}
    for row in matrix.values():
        for label, count in row.items():
            col_totals[label] = col_totals.get(label, 0) + count
    expected = sum(
        (row_totals[label] / total) * (col_totals.get(label, 0) / total) for label in row_totals
    )
    return kappa_from_margins(observed, expected)
