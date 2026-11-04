"""Infrastructure-independent orchestration for deterministic retrieval evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.fusion import RankedResultRow, reciprocal_rank_fusion

from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from .schema import GoldenQuery


class AsyncRetriever(Protocol):
    """Minimum retrieval behavior needed by the evaluation core."""

    def __call__(self, query: GoldenQuery) -> Awaitable[Sequence[str]]: ...


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    precision_at_k: float


class BranchContribution(BaseModel):
    """One branch's ranked identifiers for a single query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    branch: str
    ranked_chunk_ids: list[str]


class BranchAwareRetrieval(BaseModel):
    """A retrieval result that keeps its per-branch ranked legs.

    ``branches`` must be ordered consistently for every query so the aggregate
    attribution rows line up; the fused ranking is the one the metrics score.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fused_chunk_ids: list[str]
    branches: list[BranchContribution] = Field(default_factory=list)


class BranchAttribution(BaseModel):
    """Aggregate leave-one-branch-out attribution for one branch.

    ``aggregates_without_branch`` re-fuses every other branch (same RRF
    parameters) and re-scores the corpus; the deltas are
    ``aggregate - aggregate_without_branch``, so a branch that contributed
    nothing measures zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    branch: str
    identifiers_contributed: int
    ranked_chunk_ids: list[str]
    aggregates_without_branch: RetrievalMetrics
    delta_recall_at_k: float
    delta_reciprocal_rank: float
    delta_ndcg_at_k: float
    delta_precision_at_k: float


class RetrievalQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    k: int
    metrics: RetrievalMetrics
    branches: list[BranchContribution] = Field(default_factory=list)


class RetrievalEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[RetrievalQueryResult]
    aggregates: RetrievalMetrics
    branch_attribution: list[BranchAttribution] = Field(default_factory=list)


def refused_rank(
    branch_ranks: Mapping[str, Sequence[str]],
    excluded_branch: str,
    k: int,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """Re-fuse every branch except ``excluded_branch`` with the run's weights.

    The production RRF implementation is reused so the ablation measures the
    same fusion the ask path runs; legs keep their input order (a deterministic
    dict), and ties break by chunk identifier exactly as in production. The
    original per-branch weights are carried into the leave-one-out fusion so a
    removed branch also removes its own weight and renormalises nothing else;
    passing ``None`` keeps the historical unweighted behaviour.
    """
    legs: list[list[RankedResultRow]] = []
    remaining_weights: list[float] = []
    for index, (name, ranks) in enumerate(branch_ranks.items()):
        if name == excluded_branch:
            continue
        legs.append(
            [
                RankedResultRow(chunk_id=chunk_id, score=0.0, rank=rank)
                for rank, chunk_id in enumerate(ranks, start=1)
            ]
        )
        if weights is not None and index < len(weights):
            remaining_weights.append(weights[index])
    if not legs:
        return []
    fused = reciprocal_rank_fusion(
        *legs,
        k=60,
        limit=k,
        weights=list(remaining_weights) if weights is not None else None,
    )
    return [item.chunk_id for item in fused]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _score(retrieved: Sequence[str], expected: set[str], cutoff: int) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_k=recall_at_k(list(retrieved), expected, cutoff),
        reciprocal_rank=reciprocal_rank(list(retrieved), expected, cutoff),
        ndcg_at_k=ndcg_at_k(list(retrieved), expected, cutoff),
        precision_at_k=precision_at_k(list(retrieved), expected, cutoff),
    )


def _build_branch_attribution(
    rows: Sequence[RetrievalQueryResult],
    per_query_branch_ranks: Sequence[dict[str, list[str]]],
    aggregates: RetrievalMetrics,
    cutoff: int,
    branch_weights: Sequence[float] | None = None,
) -> list[BranchAttribution]:
    """Leave-one-branch-out aggregate attribution across the scored rows."""
    branch_names: list[str] = []
    for ranks in per_query_branch_ranks:
        for name in ranks:
            if name not in branch_names:
                branch_names.append(name)
    attribution: list[BranchAttribution] = []
    for name in branch_names:
        without_rows: list[RetrievalMetrics] = []
        contributed = 0
        ranked_union: list[str] = []
        for row, ranks in zip(rows, per_query_branch_ranks, strict=True):
            expected = set(row.expected_chunk_ids)
            without_rows.append(
                _score(refused_rank(ranks, name, cutoff, weights=branch_weights), expected, cutoff)
            )
            contributed += len(ranks.get(name, ()))
            for chunk_id in ranks.get(name, ()):
                if chunk_id not in ranked_union:
                    ranked_union.append(chunk_id)
        aggregates_without = RetrievalMetrics(
            recall_at_k=_mean([m.recall_at_k for m in without_rows]),
            reciprocal_rank=_mean([m.reciprocal_rank for m in without_rows]),
            ndcg_at_k=_mean([m.ndcg_at_k for m in without_rows]),
            precision_at_k=_mean([m.precision_at_k for m in without_rows]),
        )
        attribution.append(
            BranchAttribution(
                branch=name,
                identifiers_contributed=contributed,
                ranked_chunk_ids=ranked_union,
                aggregates_without_branch=aggregates_without,
                delta_recall_at_k=aggregates.recall_at_k - aggregates_without.recall_at_k,
                delta_reciprocal_rank=aggregates.reciprocal_rank
                - aggregates_without.reciprocal_rank,
                delta_ndcg_at_k=aggregates.ndcg_at_k - aggregates_without.ndcg_at_k,
                delta_precision_at_k=aggregates.precision_at_k - aggregates_without.precision_at_k,
            )
        )
    return attribution


async def run_retrieval_eval(
    *,
    queries: Sequence[GoldenQuery],
    retrieve: AsyncRetriever
    | Callable[[GoldenQuery], Awaitable[Sequence[str] | BranchAwareRetrieval]],
    judge_provider: object | None = None,
    cutoff: int = 10,
    branch_weights: Sequence[float] | None = None,
) -> RetrievalEvaluation:
    """Retrieve and score queries with deterministic binary-relevance metrics.

    The retriever may return either a plain ranked identifier sequence (the
    original contract, which produces no attribution) or a
    ``BranchAwareRetrieval`` carrying the per-branch ranked legs; the second
    form additionally yields leave-one-branch-out attribution rows without
    altering any existing aggregate value.
    """
    if cutoff <= 0:
        message = "Evaluation cutoff must be positive"
        raise ValueError(message)
    # The provider is an explicit attachment seam for the later judged layer. Retrieval-only
    # evaluation deliberately never invokes or inspects it.
    _ = judge_provider
    rows: list[RetrievalQueryResult] = []
    per_query_branch_ranks: list[dict[str, list[str]]] = []
    for query in queries:
        retrieved_result = await retrieve(query)
        if isinstance(retrieved_result, BranchAwareRetrieval):
            retrieved = list(retrieved_result.fused_chunk_ids)
            per_query_branch_ranks.append(
                {
                    contribution.branch: list(contribution.ranked_chunk_ids)
                    for contribution in retrieved_result.branches
                }
            )
            branch_rows = list(retrieved_result.branches)
        else:
            retrieved = list(retrieved_result)
            per_query_branch_ranks.append({})
            branch_rows = []
        expected = set(query.expected_chunk_ids)
        metrics = _score(retrieved, expected, cutoff)
        rows.append(
            RetrievalQueryResult(
                query=query.query,
                expected_chunk_ids=query.expected_chunk_ids,
                retrieved_chunk_ids=retrieved,
                k=cutoff,
                metrics=metrics,
                branches=branch_rows,
            )
        )
    aggregates = RetrievalMetrics(
        recall_at_k=_mean([row.metrics.recall_at_k for row in rows]),
        reciprocal_rank=_mean([row.metrics.reciprocal_rank for row in rows]),
        ndcg_at_k=_mean([row.metrics.ndcg_at_k for row in rows]),
        precision_at_k=_mean([row.metrics.precision_at_k for row in rows]),
    )
    return RetrievalEvaluation(
        rows=rows,
        aggregates=aggregates,
        branch_attribution=_build_branch_attribution(
            rows, per_query_branch_ranks, aggregates, cutoff, branch_weights=branch_weights
        ),
    )
