"""Unit tests for Band D item 4 — one process-lifetime shared re-ranker.

The hybrid-retrieval-ranking capability requires exactly one re-ranking
implementation whose model loads once per process. Both ranked retrieval paths
(the retrieval-graph node and the inline `ask` loop) previously constructed a
re-ranker per call; both now resolve through `get_shared_reranker()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.shared.langgraph_layer.retrieval_kb import reranker as reranker_module
from app.shared.langgraph_layer.retrieval_kb.reranker import get_shared_reranker
from app.shared.langgraph_layer.retrieval_kb.state import RetrievedChunk

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture(autouse=True)
def _cold_singleton() -> Any:
    get_shared_reranker.cache_clear()
    yield
    get_shared_reranker.cache_clear()


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk_text=text,
        preamble="pre",
        clause_type="other",
        parent_doc_id="doc-1",
        metadata_={},
        custom_metadata={},
        score=0.0,
    )


def test_the_accessor_returns_one_shared_instance() -> None:
    assert get_shared_reranker() is get_shared_reranker()


async def test_the_model_loads_once_across_two_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions: list[str] = []

    class _FakeModel:
        def __init__(self, model_name: str) -> None:
            constructions.append(model_name)

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.9] * len(pairs)

    monkeypatch.setattr(reranker_module, "CrossEncoder", _FakeModel)
    shared = get_shared_reranker()
    first = await shared.rerank("query", [_chunk("c1", "text one")], limit=5)
    second = await shared.rerank("query", [_chunk("c2", "text two")], limit=5)

    assert constructions == [reranker_module._DEFAULT_RERANKER_MODEL]
    assert [chunk.chunk_id for chunk in first] == ["c1"]
    assert [chunk.chunk_id for chunk in second] == ["c2"]


async def test_no_per_call_construction_remains_at_the_call_sites() -> None:
    import inspect

    from app.features.documents import service as documents_service
    from app.shared.langgraph_layer.retrieval_kb import nodes as retrieval_nodes

    service_source = inspect.getsource(documents_service)
    assert "CrossEncoderReranker()" not in service_source
    node_source = inspect.getsource(retrieval_nodes.make_reranker_node)
    assert "get_shared_reranker()" in node_source
