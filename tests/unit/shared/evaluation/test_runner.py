import pytest

from app.shared.evaluation.runner import run_retrieval_eval
from app.shared.evaluation.schema import GoldenQuery


def _query(*, text: str, expected: list[str]) -> GoldenQuery:
    return GoldenQuery(
        query=text,
        expected_chunk_ids=expected,
        expected_document_ids=["document"],
        jurisdiction=None,
        document_kind="contracts",
        difficulty="easy",
        notes="test",
        awaiting_sme_expansion=True,
    )


@pytest.mark.asyncio
async def test_runner_aggregates_hand_computed_scores() -> None:
    rankings = {"first": ["a", "x"], "second": ["x", "b"]}

    async def retrieve(query: GoldenQuery) -> list[str]:
        return rankings[query.query]

    evaluation = await run_retrieval_eval(
        queries=[
            _query(text="first", expected=["a"]),
            _query(text="second", expected=["b"]),
        ],
        retrieve=retrieve,
    )

    assert evaluation.aggregates.recall_at_k == pytest.approx(1.0)
    assert evaluation.aggregates.reciprocal_rank == pytest.approx(0.75)
    assert evaluation.aggregates.precision_at_k == pytest.approx(0.5)
    assert evaluation.aggregates.ndcg_at_k == pytest.approx((1.0 + 1 / 1.584962500721156) / 2)


@pytest.mark.asyncio
async def test_retrieval_only_run_never_calls_provider_double() -> None:
    class ProviderDouble:
        def __call__(self) -> None:
            message = "retrieval-only evaluation called a model provider"
            raise AssertionError(message)

    provider = ProviderDouble()

    async def retrieve(_query: GoldenQuery) -> list[str]:
        return ["a"]

    evaluation = await run_retrieval_eval(
        queries=[_query(text="first", expected=["a"])],
        retrieve=retrieve,
        judge_provider=provider,
    )

    assert evaluation.rows[0].metrics.recall_at_k == pytest.approx(1.0)
