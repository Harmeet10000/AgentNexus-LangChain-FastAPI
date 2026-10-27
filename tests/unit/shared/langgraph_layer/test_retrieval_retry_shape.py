"""Unit tests for Band D item 6 — retrieval-side retry-shape conversion.

C6 converted the ingestion callers to catch the boundary's transient type
alongside the framework type; the two retrieval callers wrapping a retried
operation were left behind with the identical defect (their degradation
branches could never fire for a wrapped transient failure). These tests drive
an exhausted retry through both converted callers and assert the degradation
branch executes — the scenario the old contract failed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.ai.graph.retry import TransientExternalError
from app.features.documents.retrieval_graph import nodes as retrieval_nodes
from app.features.documents.retrieval_graph.nodes import (
    make_graph_retrieval_node,
    make_query_analyzer_node,
)
from app.features.documents.retrieval_graph.state import QueryPlan

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import pytest


def _exhausted(label: str) -> Callable[..., Any]:
    async def _raise(*args: Any, **kwargs: Any) -> Any:
        msg = f"{label} failed after 3 attempts"
        raise TransientExternalError(msg) from ConnectionError("boom")

    return _raise


async def test_query_analyzer_degrades_on_exhausted_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval_nodes, "retry_immediate", _exhausted("gemini_query_analyzer"))

    async def _llm(*args: Any, **kwargs: Any) -> Any:
        msg = "the boundary, not the model, is under test"
        raise AssertionError(msg)

    node = make_query_analyzer_node(_llm, redis=None)
    result = await node(
        {
            "query": "who may terminate?",
            "messages": [],
            "doc_ids_filter": [],
            "iteration_count": 0,
        }
    )
    assert result["cache_hit"] is False
    assert result["query_plan"].rewritten_query == "who may terminate?"


async def test_graph_retrieval_degrades_on_exhausted_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval_nodes, "retry_immediate", _exhausted("graphiti_retrieval_search"))
    node = make_graph_retrieval_node(object())
    result = await node(
        {
            "user_id": "user-1",
            "query_plan": QueryPlan(rewritten_query="who may terminate?"),
            "doc_ids_filter": [],
        }
    )
    assert result == {"graph_chunk_ids": []}


async def test_graph_retrieval_uses_only_server_document_scope() -> None:
    calls: list[dict[str, object]] = []

    class _ScopedGraphiti:
        async def search(self, **kwargs: object) -> list[object]:
            calls.append(kwargs)
            return []

    node = make_graph_retrieval_node(_ScopedGraphiti())
    await node(
        {
            "user_id": "user-1",
            "allowed_doc_ids": ["doc-authorized"],
            "scope_bound": True,
            "query_plan": QueryPlan(rewritten_query="who may terminate?"),
            "doc_ids_filter": ["doc-authorized", "doc-requested-but-unauthorized"],
        }
    )

    assert calls == [
        {
            "query": "who may terminate?",
            "group_ids": ["doc-authorized"],
            "num_results": 20,
        }
    ]


async def test_graph_retrieval_fails_closed_without_server_scope() -> None:
    class _MustNotSearch:
        async def search(self, **_: object) -> list[object]:
            message = "graph retrieval widened an absent document scope"
            raise AssertionError(message)

    node = make_graph_retrieval_node(_MustNotSearch())
    result = await node(
        {
            "user_id": "user-1",
            "query_plan": QueryPlan(rewritten_query="who may terminate?"),
            "doc_ids_filter": ["doc-requested"],
        }
    )

    assert result == {"graph_chunk_ids": []}
