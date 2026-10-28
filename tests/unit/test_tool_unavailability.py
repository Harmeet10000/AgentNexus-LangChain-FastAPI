"""Band: agent-tools-unification group 6 — unavailability honesty.

A missing corpus must surface as an UNAVAILABLE envelope, never as a
fabricated "no results" answer and never as a rendered string.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.resilience.idempotency import IdempotencyGuard

if TYPE_CHECKING:
    from typing import Any


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
            message = "no database"
            raise SQLAlchemyError(message)

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
            message = "no db in unit test"
            raise RuntimeError(message)


async def _noop(*_a: Any, **_k: Any) -> None:
    return None


@pytest.mark.asyncio
async def test_statute_tool_answers_unavailable_when_corpus_raises(monkeypatch: Any) -> None:
    from app.features.documents.tools.retrieve_statute_section import (
        make_retrieve_statute_section_tool,
    )

    async def _raise(**_kw: Any) -> dict[str, Any]:
        message = "connection refused"
        raise SQLAlchemyError(message)

    monkeypatch.setattr(
        "app.features.documents.tools.retrieve_statute_section._fetch_statute_section",
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
    from app.features.documents.tools.search_legal_precedents import (
        make_search_legal_precedents_tool,
    )

    class _Graphiti:
        async def search_for_precedent_chains(self, **_kw: Any) -> list[Any]:
            return []

    async def _raise(**_kw: Any) -> list[dict[str, Any]]:
        message = "schema missing"
        raise SQLAlchemyError(message)

    tool = make_search_legal_precedents_tool(
        _Graphiti(),
        object(),
        _idempotency(),
    )
    monkeypatch.setattr(
        "app.features.documents.tools.search_legal_precedents._search_statutes_postgres",
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


async def test_a_partial_source_set_sets_basis_unknown_and_keeps_the_survivor(
    monkeypatch: Any,
) -> None:
    """One leg unreachable + one leg answered → ok envelope, basis_unknown=True,
    insufficient_basis withheld (None), survivor results retained."""
    from app.features.documents.tools.search_legal_precedents import (
        make_search_legal_precedents_tool,
    )

    class _Graphiti:
        async def search_for_precedent_chains(self, **_kw: Any) -> list[Any]:
            return [SimpleNamespace(name="p", content="c", relevance_score=0.9, uuid="u1")]

    async def _raise(**_kw: Any) -> list[dict[str, Any]]:
        message = "schema missing"
        raise SQLAlchemyError(message)

    tool = make_search_legal_precedents_tool(_Graphiti(), object(), _idempotency())
    monkeypatch.setattr(
        "app.features.documents.tools.search_legal_precedents._search_statutes_postgres",
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


async def test_graphiti_outage_is_not_rendered_as_empty_precedent_success(
    monkeypatch: Any,
) -> None:
    from app.features.documents.tools.search_legal_precedents import (
        make_search_legal_precedents_tool,
    )
    from app.retrieval.graphiti.schemas import GraphitiAvailability, GraphitiSearchOutcome

    class _Graphiti:
        async def search_for_precedent_chains(self, **_kw: Any) -> GraphitiSearchOutcome:
            return GraphitiSearchOutcome(
                availability=GraphitiAvailability.UNAVAILABLE,
                coverage={"requested": 5, "returned": 0},
            )

    async def _statutes(**_kw: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(
        "app.features.documents.tools.search_legal_precedents._search_statutes_postgres",
        _statutes,
    )

    tool = make_search_legal_precedents_tool(_Graphiti(), object(), _idempotency())
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
    assert result["unavailable"] is True
