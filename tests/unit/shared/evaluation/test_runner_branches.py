"""Per-branch attribution in the retrieval runner (retrieval-measurement-gate)."""

import pytest

from app.ai.evaluation.deterministic import chunks_for_page, fuse_branch_ranks
from app.ai.evaluation.runner import (
    BranchAwareRetrieval,
    BranchContribution,
    refused_rank,
    run_retrieval_eval,
)
from app.ai.evaluation.schema import GoldenQuery


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


def _branch_aware(rankings: dict[str, list[str]]):
    async def retrieve(query: GoldenQuery) -> BranchAwareRetrieval:
        branch_ranks = rankings[query.query]
        return BranchAwareRetrieval(
            fused_chunk_ids=fuse_branch_ranks(branch_ranks, k=10),
            branches=[
                BranchContribution(branch=name, ranked_chunk_ids=ranks)
                for name, ranks in branch_ranks.items()
            ],
        )

    return retrieve


@pytest.mark.asyncio
async def test_plain_retriever_produces_no_attribution_and_unchanged_metrics() -> None:
    rankings = {"first": ["a", "x"], "second": ["x", "b"]}

    async def retrieve(query: GoldenQuery) -> list[str]:
        return rankings[query.query]

    evaluation = await run_retrieval_eval(
        queries=[
            _query(text="first", expected=["a"]),
            _query(text="second", expected=["b"]),
        ],
        retrieve=retrieve,
        cutoff=2,
    )

    # The historical aggregates are untouched by the attribution feature.
    assert evaluation.aggregates.recall_at_k == pytest.approx(1.0)
    assert evaluation.aggregates.reciprocal_rank == pytest.approx(0.75)
    assert evaluation.aggregates.precision_at_k == pytest.approx(0.5)
    assert evaluation.aggregates.ndcg_at_k == pytest.approx((1.0 + 1 / 1.584962500721156) / 2)
    assert evaluation.branch_attribution == []
    assert all(row.branches == [] for row in evaluation.rows)


@pytest.mark.asyncio
async def test_branch_aware_input_matches_plain_input_aggregates() -> None:
    rankings = {
        "first": {"bm25": ["a", "x"], "trigram": ["x"]},
        "second": {"bm25": ["x"], "trigram": ["b"]},
    }
    queries = [_query(text="first", expected=["a"]), _query(text="second", expected=["b"])]

    async def plain(query: GoldenQuery) -> list[str]:
        return fuse_branch_ranks(rankings[query.query], k=2)

    evaluation_plain = await run_retrieval_eval(queries=queries, retrieve=plain, cutoff=2)
    evaluation_branches = await run_retrieval_eval(
        queries=queries, retrieve=_branch_aware(rankings), cutoff=2
    )

    assert evaluation_branches.aggregates == evaluation_plain.aggregates
    assert [row.retrieved_chunk_ids for row in evaluation_branches.rows] == [
        row.retrieved_chunk_ids for row in evaluation_plain.rows
    ]
    assert {
        contribution.branch for row in evaluation_branches.rows for contribution in row.branches
    } == {
        "bm25",
        "trigram",
    }


@pytest.mark.asyncio
async def test_branch_attribution_deltas_vanish_when_branch_contributes_nothing() -> None:
    # The trigram leg contributes identifiers the bm25 leg already contains, so
    # removing it cannot change the fused ranking for a single-query corpus.
    rankings = {"only": {"bm25": ["a", "x"], "trigram": ["a"]}}
    evaluation = await run_retrieval_eval(
        queries=[_query(text="only", expected=["a"])],
        retrieve=_branch_aware(rankings),
        cutoff=2,
    )

    assert len(evaluation.branch_attribution) == 2
    by_branch = {item.branch: item for item in evaluation.branch_attribution}
    assert by_branch["trigram"].delta_recall_at_k == pytest.approx(0.0)
    assert by_branch["trigram"].delta_ndcg_at_k == pytest.approx(0.0)
    assert by_branch["bm25"].delta_recall_at_k == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_branch_attribution_deltas_positive_for_unique_contribution() -> None:
    rankings = {"only": {"bm25": ["x", "y"], "trigram": ["a", "y"]}}
    evaluation = await run_retrieval_eval(
        queries=[_query(text="only", expected=["a"])],
        retrieve=_branch_aware(rankings),
        cutoff=3,
    )

    by_branch = {item.branch: item for item in evaluation.branch_attribution}
    # Only the trigram leg surfaces the expected identifier; removing it drops recall.
    assert by_branch["trigram"].delta_recall_at_k > 0.0
    assert by_branch["trigram"].aggregates_without_branch.recall_at_k == pytest.approx(0.0)
    assert by_branch["trigram"].identifiers_contributed == 2
    assert by_branch["bm25"].delta_recall_at_k == pytest.approx(0.0)


def test_refused_rank_re_fuses_all_but_excluded_branch() -> None:
    ranks = {"bm25": ["a", "b"], "trigram": ["b", "c"], "graph": ["c"]}
    assert refused_rank(ranks, "graph", k=3) == ["b", "a", "c"]
    # Without bm25, "c" outranks "b" (two legs vs one) and "a" vanishes entirely.
    assert refused_rank(ranks, "bm25", k=3) == ["c", "b"]
    assert refused_rank({"bm25": ["a"]}, "bm25", k=3) == []


def test_chunks_for_page_matches_the_documented_chunking_convention() -> None:
    page = "Para one line.\n\nPara two line.\n\n\n\nPara three line.\n"
    assert chunks_for_page(page) == ["Para one line.", "Para two line.", "Para three line."]
