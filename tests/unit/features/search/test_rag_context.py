"""RAG context assembly: document grouping, adjacency merge, token budget.

The budget is enforced through the injected ``CountTokens`` callable, so these
tests inject a deterministic whitespace counter instead of depending on a
tokenizer's exact counts — the budget *mechanism* is the contract, not any
particular tokenizer's numbers.
"""

from __future__ import annotations

from app.features.search.application.context_assembly import (
    SearchChunkRecord,
    assemble_rag_context,
)
from app.retrieval.fusion import RankedChunk


def _chunk(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    content: str,
) -> tuple[RankedChunk, SearchChunkRecord]:
    return (
        RankedChunk(chunk_id=chunk_id, score=1.0, rank=1),
        SearchChunkRecord(
            document_id=document_id,
            title=f"title-{document_id}",
            content=content,
            chunk_index=chunk_index,
            chunk_metadata={},
        ),
    )


def _whitespace_tokens(text: str) -> int:
    return len(text.split())


def test_adjacent_chunks_merge_into_one_document_ordered_section() -> None:
    pairs = [
        _chunk("c2", "doc-a", 1, "second part"),
        _chunk("c1", "doc-a", 0, "first part"),
    ]
    sections = assemble_rag_context(
        [rank for rank, _ in pairs],
        {rank.chunk_id: record for rank, record in pairs},
        max_tokens=100,
        count_tokens=_whitespace_tokens,
    )
    assert len(sections) == 1
    assert sections[0].document_id == "doc-a"
    assert sections[0].chunk_indices == [0, 1]
    assert sections[0].content == "first part\n\nsecond part"


def test_non_adjacent_chunks_stay_separate_sections() -> None:
    pairs = [
        _chunk("c0", "doc-a", 0, "alpha"),
        _chunk("c2", "doc-a", 2, "gamma"),
    ]
    sections = assemble_rag_context(
        [rank for rank, _ in pairs],
        {rank.chunk_id: record for rank, record in pairs},
        max_tokens=100,
        count_tokens=_whitespace_tokens,
    )
    assert [section.chunk_indices for section in sections] == [[0], [2]]


def test_documents_keep_first_hit_order() -> None:
    pairs = [
        _chunk("b1", "doc-b", 0, "b content"),
        _chunk("a1", "doc-a", 0, "a content"),
    ]
    sections = assemble_rag_context(
        [rank for rank, _ in pairs],
        {rank.chunk_id: record for rank, record in pairs},
        max_tokens=100,
        count_tokens=_whitespace_tokens,
    )
    assert [section.document_id for section in sections] == ["doc-b", "doc-a"]


def test_token_budget_truncates_and_keeps_earlier_sections() -> None:
    pairs = [
        _chunk("a1", "doc-a", 0, "one two three"),
        _chunk("b1", "doc-b", 0, "four five six seven"),
    ]
    sections = assemble_rag_context(
        [rank for rank, _ in pairs],
        {rank.chunk_id: record for rank, record in pairs},
        max_tokens=5,
        count_tokens=_whitespace_tokens,
    )
    # doc-a's 3 tokens fit; doc-b's 4 would push the total past 5.
    assert [section.document_id for section in sections] == ["doc-a"]


def test_budget_exactly_reached_admits_the_section() -> None:
    pairs = [_chunk("a1", "doc-a", 0, "one two three")]
    sections = assemble_rag_context(
        [pairs[0][0]],
        {pairs[0][0].chunk_id: pairs[0][1]},
        max_tokens=3,
        count_tokens=_whitespace_tokens,
    )
    assert len(sections) == 1


def test_merged_group_over_budget_keeps_the_fitting_prefix() -> None:
    """An overflowing merged run is split, not dropped wholesale (G P2)."""
    pairs = [
        _chunk("a1", "doc-a", 0, "one two"),  # merged with a2 = 4 tokens
        _chunk("a2", "doc-a", 1, "three four"),
    ]
    sections = assemble_rag_context(
        [rank for rank, _ in pairs],
        {rank.chunk_id: record for rank, record in pairs},
        max_tokens=3,
        count_tokens=_whitespace_tokens,
    )
    # The merged pair (4) overflows 3, so the run is emitted as its largest
    # fitting prefix — chunk 0 alone (2 tokens) — and chunk 1 cannot fit alone.
    assert [section.chunk_indices for section in sections] == [[0]]
    assert sections[0].content == "one two"


def test_overflow_chunk_starts_the_next_section_when_it_fits() -> None:
    """A separator-costing counter makes merged content exceed the token sum;
    the overflow chunk must then start its own section, not vanish with the
    group."""

    def _count_with_separator_cost(text: str) -> int:
        return len(text.split()) + (1 if "\n\n" in text else 0)

    pairs = [
        _chunk("a1", "doc-a", 0, "one"),  # 1 token
        _chunk("a2", "doc-a", 1, "two"),  # merged = 1+1+1(separator) = 3
    ]
    sections = assemble_rag_context(
        [rank for rank, _ in pairs],
        {rank.chunk_id: record for rank, record in pairs},
        max_tokens=2,
        count_tokens=_count_with_separator_cost,
    )
    # a1+a2 merged (3) overflows; a1 alone fits, and a2 alone (1 ≤ remaining 1)
    # starts a new section rather than being dropped with the group.
    assert [section.chunk_indices for section in sections] == [[0], [1]]


def test_group_whose_first_chunk_cannot_fit_is_skipped_but_later_groups_survive() -> None:
    pairs = [
        _chunk("big", "doc-a", 0, " ".join(["token"] * 10)),  # 10 tokens, never fits
        _chunk("b1", "doc-b", 0, "tiny"),  # 1 token, fits
    ]
    sections = assemble_rag_context(
        [rank for rank, _ in pairs],
        {rank.chunk_id: record for rank, record in pairs},
        max_tokens=5,
        count_tokens=_whitespace_tokens,
    )
    assert [section.document_id for section in sections] == ["doc-b"]


def test_unknown_chunk_ids_are_skipped() -> None:
    sections = assemble_rag_context(
        [RankedChunk(chunk_id="missing", score=1.0, rank=1)],
        {},
        max_tokens=100,
        count_tokens=_whitespace_tokens,
    )
    assert sections == []
