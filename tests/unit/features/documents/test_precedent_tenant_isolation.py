"""V1 two-user precedent regression (tasks 1.1 + 2.3).

Two users in different tenants store identical statute-like chunks. The
Postgres precedent leg must bind every ranked and detail query to the
caller's server-derived tenant and the explicit document allowlist, so an
identical chunk belonging to another user or tenant can never enter the
caller's result set — no matter what the corpus contains.

There is no live Postgres in the unit band, so the leak is pinned at the
strongest CI-runnable layer: every statement executed for user A carries
only A's tenant and A's allowlist as binds, and the same statements
executed for user B carry only B's. A predicate regression (the historical
V1 defect — user_id-free unscoped statute SQL) fails these assertions.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain.tools import ToolRuntime
from sqlalchemy import TextClause
from sqlalchemy.dialects import postgresql

from app.features.documents.application.tools.search_legal_precedents import (
    _search_statutes_postgres,
    make_search_legal_precedents_tool,
)
from app.resilience.idempotency import IdempotencyGuard
from app.retrieval.graphiti.schemas import GraphitiSearchOutcome

if TYPE_CHECKING:
    from typing import Any

_ALICE_TENANT = "personal:alice"
_BOB_TENANT = "personal:bob"
_ALICE_DOCS = ("11111111-1111-1111-1111-111111111111",)
_BOB_DOCS = ("22222222-2222-2222-2222-222222222222",)
# Both tenants ingest the exact same statute-like chunk text; only the
# owning tenant differs. This is the V1 leak precondition.
_IDENTICAL_STATUTE = (
    "Section 73: When a contract has been broken, the party who suffers "
    "from the breach is entitled to receive compensation."
)
_IDENTICAL_CHUNK_ID = "99999999-9999-9999-9999-999999999999"


class _Rows:
    """Result proxy: ranked queries return (chunk_id, rank); detail returns full rows."""

    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None


class _CaptureConnection:
    def __init__(self, calls: list[tuple[Any, Any]]) -> None:
        self._calls = calls

    async def execute(self, statement: Any, params: Any = None) -> _Rows:
        self._calls.append((statement, params))
        if isinstance(statement, TextClause):
            # Ranked branches: both resolve the SAME identical chunk id, as a
            # real shared corpus would — isolation must come from the binds.
            return _Rows([(_IDENTICAL_CHUNK_ID, 1)])
        return _Rows(
            [
                (
                    _IDENTICAL_CHUNK_ID,
                    "Indian Contract Act 1872",
                    "73",
                    _IDENTICAL_STATUTE,
                    _ALICE_DOCS[0],
                    "Indian Contract Act 1872",
                    1872,
                )
            ]
        )

    async def __aenter__(self) -> _CaptureConnection:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _CaptureEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def connect(self) -> _CaptureConnection:
        return _CaptureConnection(self.calls)


def _statement_binds(statement: Any, params: Any) -> dict[str, Any]:
    """Extract the concrete bind values a statement executes with."""
    if isinstance(statement, TextClause):
        return dict(params or {})
    return dict(statement.compile(dialect=postgresql.dialect()).params)


def _scope_binds(calls: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
    """Tenant/allowlist bind values for every executed statement.

    ``text()`` branches name their binds explicitly; the ORM detail query
    auto-names them (``tenant_id_1``), so the values are matched positionally
    rather than by key.
    """
    scoped: list[dict[str, Any]] = []
    for statement, params in calls:
        binds = _statement_binds(statement, params)
        scoped.append(
            {
                "tenant": next(
                    (value for value in binds.values() if isinstance(value, str) and ":" in value),
                    None,
                ),
                "allowlists": [
                    tuple(value)
                    for value in binds.values()
                    if isinstance(value, (list, tuple, set, frozenset))
                ],
            }
        )
    return scoped


def _null_idempotency() -> IdempotencyGuard:
    from unittest.mock import MagicMock

    redis = MagicMock()

    async def _none(*_a: Any, **_k: Any) -> None:
        return None

    redis.get = _none
    redis.set = _none

    class _Unreachable:
        def connect(self) -> None:
            message = "no database in unit test"
            raise RuntimeError(message)

    return IdempotencyGuard(redis=redis, db_engine=_Unreachable())


def _runtime(tenant_id: str, allowed_doc_ids: tuple[str, ...]) -> ToolRuntime:
    """Server-created ToolRuntime carrying tenant A's or B's scope."""
    return ToolRuntime(
        state={},
        context={
            "user_id": tenant_id.removeprefix("personal:"),
            "tenant_id": tenant_id,
            "allowed_doc_ids": allowed_doc_ids,
            "allowed_operations": ("evidence.search",),
            "thread_id": "thread-1",
            "current_step": "1",
            "run_id": "run-1",
            "checkpoint_id": "cp-1",
            "document_versions": (),
            "document_contracts": (),
            "capabilities": (),
        },
        config={},
        stream_writer=None,
        tool_call_id="test-call",
        store=None,
    )


class _CaptureGraphiti:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def search_for_precedent_chains(self, **kwargs: Any) -> GraphitiSearchOutcome:
        self.calls.append(kwargs)
        return GraphitiSearchOutcome()


@pytest.mark.asyncio
async def test_identical_statute_chunks_resolve_only_through_the_callers_scope() -> None:
    """The V1 leak is closed at the bind level: user A's queries bind A's
    tenant and A's allowlist only, even though both tenants hold the same
    chunk and the corpus resolves the same chunk id for both callers."""
    alice_engine = _CaptureEngine()
    alice_rows = await _search_statutes_postgres(
        db_engine=alice_engine,
        query="compensation for breach",
        jurisdiction="India",
        limit=5,
        tenant_id=_ALICE_TENANT,
        allowed_doc_ids=_ALICE_DOCS,
    )
    bob_engine = _CaptureEngine()
    await _search_statutes_postgres(
        db_engine=bob_engine,
        query="compensation for breach",
        jurisdiction="India",
        limit=5,
        tenant_id=_BOB_TENANT,
        allowed_doc_ids=_BOB_DOCS,
    )

    # Every statement (bm25, trigram, fused detail) carried A's scope for A
    # and B's scope for B — an identical chunk in the other tenant is
    # unreachable because it is outside the bound tenant/allowlist.
    assert alice_rows, "the fixture must produce a result for the authorized caller"
    for scoped in _scope_binds(alice_engine.calls):
        assert scoped["tenant"] == _ALICE_TENANT
        assert _ALICE_DOCS in scoped["allowlists"]
        assert _BOB_DOCS not in scoped["allowlists"]
    for scoped in _scope_binds(bob_engine.calls):
        assert scoped["tenant"] == _BOB_TENANT
        assert _BOB_DOCS in scoped["allowlists"]
        assert _ALICE_DOCS not in scoped["allowlists"]

    # The predicate itself is structural, not incidental: tenant and
    # allowlist filters appear in the compiled SQL of every statement.
    for statement, _params in alice_engine.calls:
        compiled = (
            statement
            if isinstance(statement, TextClause)
            else statement.compile(dialect=postgresql.dialect())
        )
        sql = str(compiled)
        assert "tenant_id" in sql, "every precedent query must carry the tenant predicate"
        assert "document_id" in sql or "allowed_doc_ids" in sql, (
            "every precedent query must carry the allowlist predicate"
        )


@pytest.mark.asyncio
async def test_empty_allowlist_fails_closed_without_touching_the_corpus() -> None:
    engine = _CaptureEngine()
    rows = await _search_statutes_postgres(
        db_engine=engine,
        query="compensation for breach",
        jurisdiction="India",
        limit=5,
        tenant_id=_ALICE_TENANT,
        allowed_doc_ids=(),
    )
    assert rows == []
    assert engine.calls == [], "an empty allowlist must not issue any corpus read"


@pytest.mark.asyncio
async def test_precedent_tool_derives_scope_from_runtime_never_from_the_model() -> None:
    """Tool-level V1 check: the same identical-corpus query issued by Alice
    and by Bob scopes the Graphiti leg AND the Postgres leg to each caller's
    own tenant; neither invocation can bind the other's scope."""
    graphiti = _CaptureGraphiti()
    tool = make_search_legal_precedents_tool(graphiti, object(), _null_idempotency())

    async def _scoped_search(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": _IDENTICAL_CHUNK_ID,
                "title": "Indian Contract Act 1872",
                "section_ref": "73",
                "excerpt": _IDENTICAL_STATUTE,
                "jurisdiction": None,
                "act_name": "Indian Contract Act 1872",
                "year": 1872,
                "rank": 1.0,
                "source": "unified_corpus",
            }
        ]

    # Bypass the raw-SQL engine double here: the bind-level coverage lives in
    # the statement-capture tests above. The tool-level check is about whose
    # scope reaches the legs.
    import app.features.documents.application.tools.search_legal_precedents as precedent_module

    original = precedent_module._search_statutes_postgres
    engines: list[_CaptureEngine] = []

    async def _capturing_search(**kwargs: Any) -> list[dict[str, Any]]:
        engines.append(
            SimpleNamespace(
                tenant_id=kwargs.get("tenant_id"),
                allowed_doc_ids=kwargs.get("allowed_doc_ids"),
            )
        )
        return await _scoped_search(**kwargs)

    precedent_module._search_statutes_postgres = _capturing_search  # type: ignore[method-assign]
    try:
        alice_result = await tool.ainvoke(
            {
                "query": "compensation for breach",
                "clause_id": "c-1",
                "jurisdiction": "India",
                "runtime": _runtime(_ALICE_TENANT, _ALICE_DOCS),
            }
        )
        bob_result = await tool.ainvoke(
            {
                "query": "compensation for breach",
                "clause_id": "c-1",
                "jurisdiction": "India",
                "runtime": _runtime(_BOB_TENANT, _BOB_DOCS),
            }
        )
    finally:
        precedent_module._search_statutes_postgres = original  # type: ignore[method-assign]

    assert not isinstance(alice_result, str)
    assert not isinstance(bob_result, str)
    assert alice_result["success"] is True
    assert bob_result["success"] is True

    # Graphiti leg: each caller reached only their own tenant/allowlist.
    assert [call["tenant_id"] for call in graphiti.calls] == [_ALICE_TENANT, _BOB_TENANT]
    assert [tuple(call["allowed_doc_ids"]) for call in graphiti.calls] == [
        _ALICE_DOCS,
        _BOB_DOCS,
    ]
    # Postgres leg: same per-caller scoping.
    assert [call.tenant_id for call in engines] == [_ALICE_TENANT, _BOB_TENANT]
    assert [tuple(call.allowed_doc_ids) for call in engines] == [_ALICE_DOCS, _BOB_DOCS]

    # Statute rows expose no jurisdiction of their own: the corpus is
    # jurisdiction-agnostic today, so "applicable jurisdiction filtering" on
    # this leg is a recorded no-op (see the change notes for task 2.3).
    for statute in alice_result["data"]["statutes"]:
        assert statute["jurisdiction"] is None
