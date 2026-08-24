"""Band: agent-tools-unification group 6 — unavailability honesty.

A missing corpus must surface as an UNAVAILABLE envelope, never as a
fabricated "no results" answer and never as a rendered string.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.shared.langchain_layer.agents.tools.idempotency import IdempotencyGuard


async def _async_none(*_a: Any, **_k: Any) -> None:
    return None


def _idempotency() -> Any:
    """A real IdempotencyGuard whose cache layers always miss."""
    from unittest.mock import MagicMock

    redis = MagicMock()
    redis.get = _async_none
    redis.set = _async_none

    class _FailingEngine:
        """connect() fails like an unreachable database."""

        def connect(self):
            raise SQLAlchemyError("no database")

    return IdempotencyGuard(redis=redis, db_engine=_FailingEngine())


class _NullRedis:
    async def get(self, _key: str) -> None:
        return None

    async def set(self, *a: Any, **_kw: Any) -> None:
        return None


class _NullDB:
    class engine:  # noqa: N801 — attribute holder only
        @staticmethod
        def connect():
            raise RuntimeError("no db in unit test")


async def _noop(*_a: Any, **_k: Any) -> None:
    return None


@pytest.mark.asyncio
async def test_statute_tool_answers_unavailable_when_corpus_raises(monkeypatch: Any) -> None:
    from app.shared.langchain_layer.agents.tools.retrieve_statute_section import (
        make_retrieve_statute_section_tool,
    )

    async def _raise(**_kw: Any) -> dict[str, Any]:
        raise SQLAlchemyError("connection refused")

    monkeypatch.setattr(
        "app.shared.langchain_layer.agents.tools.retrieve_statute_section._fetch_statute_section",
        _raise,
    )
    tool = make_retrieve_statute_section_tool(
        db_engine=object(),
        idempotency=_idempotency(),
    )
    result = await tool.ainvoke(
        {
            "act_name": "Contract Act",
            "section_ref": "73",
            "jurisdiction": "India",
            "user_id": "u",
            "thread_id": "t",
            "step_id": "s",
        }
    )
    assert not isinstance(result, str), "an envelope must not be rendered into a sentence"
    assert result["unavailable"] is True
    assert result["error"]


@pytest.mark.asyncio
async def test_precedent_search_answers_unavailable_when_corpus_raises(monkeypatch: Any) -> None:
    from app.shared.langchain_layer.agents.tools.search_legal_precedents import (
        make_search_legal_precedents_tool,
    )

    class _Graphiti:
        async def search_for_precedent_chains(self, **_kw: Any) -> list[Any]:
            return []

    async def _raise(**_kw: Any) -> list[dict[str, Any]]:
        raise SQLAlchemyError("schema missing")

    tool = make_search_legal_precedents_tool(
        _Graphiti(),
        object(),
        _idempotency(),
    )
    monkeypatch.setattr(
        "app.shared.langchain_layer.agents.tools.search_legal_precedents._search_statutes_postgres",
        _raise,
    )
    result = await tool.ainvoke(
        {
            "query": "penalty clauses",
            "clause_id": "c1",
            "jurisdiction": "India",
            "user_id": "u",
            "thread_id": "t",
            "step_id": "s",
        }
    )
    assert not isinstance(result, str)
    assert result["unavailable"] is True


@pytest.mark.asyncio
async def test_hybrid_tool_reports_the_vector_layer_unavailable() -> None:
    from app.shared.langchain_layer.agents.tools.precedent_tools import (
        make_hybrid_retrieve_precedents_tool,
    )

    class _Graphiti:
        async def search_for_precedent_chains(self, **_kw: Any) -> list[Any]:
            return []

    class _EmptySubgraph:
        nodes: list = []
        edges: list = []

        def to_context_text(self) -> str:
            return ""

    class _Expander:
        async def expand_from_seeds(self, **_kw: Any) -> Any:
            return _EmptySubgraph()

    tool = make_hybrid_retrieve_precedents_tool(
        _Graphiti(),
        _Expander(),
        object(),
        _idempotency(),
    )
    result = await tool.ainvoke(
        {
            "query": "q",
            "user_id": "u",
            "doc_id": "d",
            "num_results": 5,
            "thread_id": "t",
            "step_id": "s",
        }
    )
    assert not isinstance(result, str)
    assert result["unavailable"] is True
    assert "pgvector" in result["error"]


async def test_a_partial_source_set_sets_basis_unknown_and_keeps_the_survivor(
    monkeypatch: Any,
) -> None:
    """One leg unreachable + one leg answered → ok envelope, basis_unknown=True,
    insufficient_basis withheld (None), survivor results retained."""
    from app.shared.langchain_layer.agents.tools.search_legal_precedents import (
        make_search_legal_precedents_tool,
    )

    class _Graphiti:
        async def search_for_precedent_chains(self, **_kw: Any) -> list[Any]:
            return [SimpleNamespace(name="p", content="c", relevance_score=0.9, uuid="u1")]

    async def _raise(**_kw: Any) -> list[dict[str, Any]]:
        raise SQLAlchemyError("schema missing")

    tool = make_search_legal_precedents_tool(_Graphiti(), object(), _idempotency())
    monkeypatch.setattr(
        "app.shared.langchain_layer.agents.tools.search_legal_precedents._search_statutes_postgres",
        _raise,
    )
    result = await tool.ainvoke(
        {
            "query": "penalty clauses",
            "clause_id": "c1",
            "jurisdiction": "India",
            "user_id": "u",
            "thread_id": "t",
            "step_id": "s",
        }
    )
    assert not isinstance(result, str)
    assert result["success"] is True, "the surviving source must still be returned"
    assert result["data"]["basis_unknown"] is True
    assert result["data"]["insufficient_basis"] is None
    assert result["data"]["unavailable_layers"] == ["statutes"]
    assert len(result["data"]["precedents"]) == 1
