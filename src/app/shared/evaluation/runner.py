"""Infrastructure-independent orchestration for deterministic retrieval evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict

from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

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


class RetrievalQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    k: int
    metrics: RetrievalMetrics


class RetrievalEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: list[RetrievalQueryResult]
    aggregates: RetrievalMetrics


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def run_retrieval_eval(
    *,
    queries: Sequence[GoldenQuery],
    retrieve: AsyncRetriever | Callable[[GoldenQuery], Awaitable[Sequence[str]]],
    judge_provider: object | None = None,
) -> RetrievalEvaluation:
    """Retrieve and score queries with deterministic binary-relevance metrics."""
    # The provider is an explicit attachment seam for the later judged layer. Retrieval-only
    # evaluation deliberately never invokes or inspects it.
    _ = judge_provider
    rows: list[RetrievalQueryResult] = []
    for query in queries:
        retrieved = list(await retrieve(query))
        k = max(len(retrieved), 1)
        expected = set(query.expected_chunk_ids)
        metrics = RetrievalMetrics(
            recall_at_k=recall_at_k(retrieved, expected, k),
            reciprocal_rank=reciprocal_rank(retrieved, expected, k),
            ndcg_at_k=ndcg_at_k(retrieved, expected, k),
            precision_at_k=precision_at_k(retrieved, expected, k),
        )
        rows.append(
            RetrievalQueryResult(
                query=query.query,
                expected_chunk_ids=query.expected_chunk_ids,
                retrieved_chunk_ids=retrieved,
                k=k,
                metrics=metrics,
            )
        )
    return RetrievalEvaluation(
        rows=rows,
        aggregates=RetrievalMetrics(
            recall_at_k=_mean([row.metrics.recall_at_k for row in rows]),
            reciprocal_rank=_mean([row.metrics.reciprocal_rank for row in rows]),
            ndcg_at_k=_mean([row.metrics.ndcg_at_k for row in rows]),
            precision_at_k=_mean([row.metrics.precision_at_k for row in rows]),
        ),
    )
