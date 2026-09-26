"""Post-processing: dedupe by identity, restore document order (task 4.1)."""

from __future__ import annotations

from app.shared.langgraph_layer.retrieval_kb.nodes import make_post_process_node
from app.shared.langgraph_layer.retrieval_kb.state import RetrievedChunk


def _chunk(chunk_id: str, doc: str, index: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk_text=f"text {chunk_id}",
        preamble="pre",
        clause_type="other",
        parent_doc_id=doc,
        chunk_index=index,
        metadata_={},
        custom_metadata={},
        score=score,
    )


def _words(text: str) -> int:
    return len(text.split())


async def test_overlapping_chunk_appears_once_in_document_order() -> None:
    node = make_post_process_node(max_tokens=10_000, count_tokens=_words)
    # Two branches returned chunk b twice (relevance order: b, a, b, c);
    # document order within doc-1 is a(0), b(1).
    reranked = [
        _chunk("b", "doc-1", 1, 0.95),
        _chunk("a", "doc-1", 0, 0.90),
        _chunk("b", "doc-1", 1, 0.95),
        _chunk("c", "doc-2", 0, 0.80),
    ]

    result = await node({"reranked_chunks": reranked})

    assert [chunk.chunk_id for chunk in result["reranked_chunks"]] == ["a", "b", "c"]
    assert result["assembled_context"]
    assert result["reranked_chunks"][0].chunk_index == 0


async def test_empty_reranked_list_assembles_nothing() -> None:
    node = make_post_process_node(max_tokens=10_000, count_tokens=_words)

    result = await node({"reranked_chunks": []})

    assert result["reranked_chunks"] == []
    assert result["assembled_context"] == []


async def test_token_budget_removes_chunks_from_llm_input() -> None:
    node = make_post_process_node(max_tokens=2, count_tokens=_words)
    reranked = [
        _chunk("a", "doc-1", 0, 0.9),
        _chunk("b", "doc-2", 0, 0.8),
    ]

    result = await node({"reranked_chunks": reranked})

    assert [chunk.chunk_id for chunk in result["reranked_chunks"]] == ["a"]
    assert [section.document_id for section in result["assembled_context"]] == ["doc-1"]
