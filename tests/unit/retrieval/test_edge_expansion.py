"""Unit tests for the pure edge-aware context expansion core."""

from __future__ import annotations

from app.retrieval.edge_expansion import (
    EdgeSeedChunk,
    LegalEdgeRecord,
    expand_along_edges,
    format_edge_block_text,
)


def _count_tokens(text: str) -> int:
    return len(text.split())


def _edge(
    edge_id: str = "edge-1",
    *,
    document_id: str = "doc-a",
    from_entity: str = "Acme Corp",
    relationship: str = "obliges",
    to_entity: str = "GlobalTech Ltd",
    clause_id: str = "2.1",
    char_start: int | None = 1042,
    char_end: int | None = 1110,
) -> LegalEdgeRecord:
    return LegalEdgeRecord(
        edge_id=edge_id,
        document_id=document_id,
        from_entity=from_entity,
        relationship=relationship,
        to_entity=to_entity,
        clause_id=clause_id,
        char_start=char_start,
        char_end=char_end,
        confidence=0.9,
    )


def _seed(
    chunk_id: str = "chunk-1",
    *,
    document_id: str = "doc-a",
    content: str = "The Employer shall pay the monthly charge. Acme Corp signs.",
    clause_ids: tuple[str, ...] = (),
) -> EdgeSeedChunk:
    return EdgeSeedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        clause_ids=clause_ids,
    )


async def test_expansion_cites_edge_span_and_identity() -> None:
    expansion = expand_along_edges(
        seeds=[_seed()],
        edges=[_edge()],
        max_expansion_tokens=64,
        count_tokens=_count_tokens,
    )

    assert len(expansion) == 1
    assert expansion[0].seed_chunk_id == "chunk-1"
    assert expansion[0].document_id == "doc-a"
    (block,) = expansion[0].blocks
    assert block.edge_id == "edge-1"
    assert block.char_start == 1042
    assert block.char_end == 1110
    assert "chars 1042-1110" in block.text
    assert "OBLIGES" in block.text
    assert "clause 2.1" in block.text


async def test_ungrounded_edge_cites_span_unavailable() -> None:
    expansion = expand_along_edges(
        seeds=[_seed()],
        edges=[_edge(char_start=None, char_end=None)],
        max_expansion_tokens=64,
        count_tokens=_count_tokens,
    )

    (block,) = expansion[0].blocks
    assert block.char_start is None
    assert "chars unavailable" in block.text


async def test_blocks_are_ordered_by_span_regardless_of_input_order() -> None:
    late = _edge("edge-late", clause_id="9.1", char_start=5000, char_end=5100)
    early = _edge("edge-early", clause_id="1.1", char_start=10, char_end=90)
    middle = _edge("edge-mid", clause_id="4.2", char_start=900, char_end=950)

    expansion = expand_along_edges(
        seeds=[_seed()],
        edges=[late, middle, early],
        max_expansion_tokens=512,
        count_tokens=_count_tokens,
    )

    assert [block.edge_id for block in expansion[0].blocks] == [
        "edge-early",
        "edge-mid",
        "edge-late",
    ]


async def test_token_budget_bounds_expansion_without_truncation() -> None:
    big = _edge("edge-big")
    # Span-first ordering puts `big` (chars 1042-1110) ahead of `small`
    # (chars 5000-5100), so the first block is the one consuming the budget.
    small = _edge("edge-small", clause_id="3.3", char_start=5000, char_end=5100)

    expansion = expand_along_edges(
        seeds=[_seed()],
        edges=[big, small],
        max_expansion_tokens=_count_tokens(format_edge_block_text(big)),
        count_tokens=_count_tokens,
    )

    # The first block consumed the whole budget, so the second is skipped
    # whole — never truncated — and no block exceeds the budget.
    assert [block.edge_id for block in expansion[0].blocks] == ["edge-big"]
    assert sum(block.token_count for block in expansion[0].blocks) <= _count_tokens(
        format_edge_block_text(big)
    )


async def test_smaller_block_still_fits_after_a_oversized_one_is_skipped() -> None:
    huge = _edge("edge-huge", char_start=5000, char_end=6000)
    tiny = _edge("edge-tiny", clause_id="3.1", char_start=1, char_end=2)
    tiny_text_tokens = _count_tokens(format_edge_block_text(tiny))

    expansion = expand_along_edges(
        seeds=[_seed()],
        edges=[huge, tiny],
        max_expansion_tokens=tiny_text_tokens + 1,
        count_tokens=_count_tokens,
    )

    assert [block.edge_id for block in expansion[0].blocks] == ["edge-tiny"]


async def test_edge_expands_only_the_first_matching_seed() -> None:
    first = _seed("chunk-first")
    second = _seed("chunk-second", content="GlobalTech Ltd accepts the terms.")

    expansions = expand_along_edges(
        seeds=[first, second],
        edges=[_edge()],
        max_expansion_tokens=64,
        count_tokens=_count_tokens,
    )

    assert [expansion.seed_chunk_id for expansion in expansions] == ["chunk-first"]


async def test_clause_identity_match_does_not_require_entity_occurrence() -> None:
    seed = _seed(content="Payment terms govern this schedule.", clause_ids=("2.1",))

    expansion = expand_along_edges(
        seeds=[seed],
        edges=[_edge()],
        max_expansion_tokens=64,
        count_tokens=_count_tokens,
    )

    assert expansion[0].blocks[0].clause_id == "2.1"


async def test_short_entities_never_match_by_occurrence() -> None:
    seed = _seed(content="A B C sign the deed.")
    expansion = expand_along_edges(
        seeds=[seed],
        edges=[_edge(from_entity="A", to_entity="B")],
        max_expansion_tokens=64,
        count_tokens=_count_tokens,
    )

    assert expansion == []


async def test_edges_from_other_documents_do_not_expand() -> None:
    expansion = expand_along_edges(
        seeds=[_seed()],
        edges=[_edge(document_id="doc-b")],
        max_expansion_tokens=64,
        count_tokens=_count_tokens,
    )

    assert expansion == []


async def test_matching_is_case_insensitive_and_deterministic() -> None:
    seed = _seed(content="acme corp agrees to the schedule.")
    first_run = expand_along_edges(
        seeds=[seed],
        edges=[_edge(), _edge("edge-2", clause_id="3.1", char_start=5, char_end=9)],
        max_expansion_tokens=512,
        count_tokens=_count_tokens,
    )
    second_run = expand_along_edges(
        seeds=[seed],
        edges=[_edge("edge-2", clause_id="3.1", char_start=5, char_end=9), _edge()],
        max_expansion_tokens=512,
        count_tokens=_count_tokens,
    )

    assert first_run == second_run


async def test_zero_budget_and_empty_inputs_expand_nothing() -> None:
    for kwargs in (
        {"seeds": [_seed()], "edges": [_edge()], "max_expansion_tokens": 0},
        {"seeds": [], "edges": [_edge()], "max_expansion_tokens": 64},
        {"seeds": [_seed()], "edges": [], "max_expansion_tokens": 64},
    ):
        result: list[object] = expand_along_edges(
            count_tokens=_count_tokens,
            **kwargs,
        )  # ty: ignore[invalid-argument-type]
        assert result == []
