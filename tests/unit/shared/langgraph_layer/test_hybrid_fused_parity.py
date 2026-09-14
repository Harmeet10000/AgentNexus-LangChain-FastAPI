"""Graph path and service search share one fused path (retrieval-sql task 3.2).

`make_hybrid_retrieval_node` and `DocumentQueryService.search` both run
`retrieve_fused` with the same inputs and weights, so one query must yield
identical chunk-id order from both doors. A divergence fails here rather than
serving different results depending on which door the query entered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from app.features.documents.constants import RRF_WEIGHT_TRIGRAM
from app.features.documents.dto import UnifiedSearchRequest
from app.features.documents.service import DocumentQueryService
from app.shared.langgraph_layer.retrieval_kb import nodes as retrieval_nodes
from app.shared.langgraph_layer.retrieval_kb.nodes import make_hybrid_retrieval_node
from app.shared.langgraph_layer.retrieval_kb.state import QueryPlan

if TYPE_CHECKING:
    from typing import Any

    import pytest

_ROWS = {
    "bm25": [{"chunk_id": "a", "score": -5.0}, {"chunk_id": "b", "score": -2.0}],
    "vector": [{"chunk_id": "b", "score": 0.9}, {"chunk_id": "c", "score": 0.8}],
    "trigram": [{"chunk_id": "c", "score": 0.7}, {"chunk_id": "a", "score": 0.6}],
}

_LOOKUP = {
    cid: {
        "chunk_id": cid,
        "document_id": "doc-1",
        "title": "t",
        "content": f"content {cid}",
        "preamble": "",
        "search_text": f"content {cid}",
        "chunk_index": 0,
        "chunk_kind": "generic",
        "clause_type": None,
        "chunk_metadata": {},
        "quality_warnings": [],
        "graphiti_verified": False,
    }
    for cid in ("a", "b", "c")
}


class _StubRepo:
    async def bm25_search(self, **_kwargs: Any) -> Any:
        from returns.result import Success

        return Success([dict(row) for row in _ROWS["bm25"]])

    async def vector_search(self, **_kwargs: Any) -> Any:
        from returns.result import Success

        return Success([dict(row) for row in _ROWS["vector"]])

    async def trigram_search(self, **_kwargs: Any) -> Any:
        from returns.result import Success

        return Success([dict(row) for row in _ROWS["trigram"]])

    async def fetch_chunks_by_ids(self, chunk_ids: Any) -> Any:
        from returns.result import Success

        return Success({cid: dict(_LOOKUP[cid]) for cid in chunk_ids})


async def _fake_embed(*_args: Any, **_kwargs: Any) -> list[float]:
    return [0.0, 0.1, 0.2]


async def test_graph_path_and_service_search_agree_on_chunk_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval_nodes, "embed_text", _fake_embed)
    monkeypatch.setattr(
        "app.features.documents.service.embed_text", _fake_embed
    )
    weights = [0.6, 0.4, RRF_WEIGHT_TRIGRAM]
    repo = _StubRepo()

    node = make_hybrid_retrieval_node(cast("Any", repo), redis=None)
    node_result = await node(
        {
            "user_id": "user-1",
            "query_plan": QueryPlan(rewritten_query="indemnity cap"),
            "doc_ids_filter": [],
        }
    )
    node_order = [chunk.chunk_id for chunk in node_result["retrieved_chunks"]]

    service = DocumentQueryService(
        cast("Any", repo), cast("Any", None), None, None
    )
    search_result = await service.search(
        user_id="user-1",
        payload=UnifiedSearchRequest(
            query="indemnity cap", limit=20, candidate_limit=50
        ),
        weights=weights,
    )
    from returns.result import Success as _Success

    assert isinstance(search_result, _Success)
    service_order = [item.chunk_id for item in search_result.unwrap().items]

    assert node_order == service_order == ["a", "c", "b"]
