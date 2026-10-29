"""Regression tests for server-owned tenant and document scope."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any

from app.features.auth.domain.tenancy import personal_tenant_id
from app.features.documents.repository import build_chunk_rows
from app.features.documents.tools.authorization import trusted_tool_context
from app.retrieval.graphiti.client import (
    get_obligation_chain,
    search_for_precedent_chains,
    search_for_risk_context,
)


def test_personal_tenants_are_stable_and_distinct() -> None:
    assert personal_tenant_id("alice") == "personal:alice"
    assert personal_tenant_id("alice") != personal_tenant_id("bob")


def test_chunk_rows_carry_server_owned_tenant_scope() -> None:
    rows = build_chunk_rows(
        document_id="00000000-0000-0000-0000-000000000001",
        user_id="alice",
        tenant_id="tenant-a",
        chunks=[{"content": "private clause", "chunk_index": 0}],
    )

    assert rows[0]["tenant_id"] == "tenant-a"
    assert rows[0]["user_id"] == "alice"


def test_runtime_scope_cannot_be_replaced_by_model_arguments() -> None:
    runtime = SimpleNamespace(
        state={
            "user_id": "alice",
            "tenant_id": "tenant-a",
            "allowed_doc_ids": ["doc-a"],
            "thread_id": "thread-a",
            "current_step": "retrieve",
        },
        context={},
    )

    context = trusted_tool_context(runtime)

    assert context.user_id == "alice"
    assert context.tenant_id == "tenant-a"
    assert context.allowed_doc_ids == ("doc-a",)
    context.require_document("doc-a")

    try:
        context.require_document("doc-b")
    except PermissionError:
        pass
    else:
        message = "a document outside allowed_doc_ids must be denied"
        raise AssertionError(message)


def test_tool_scope_requires_server_runtime_and_never_synthesizes_tenant() -> None:
    with pytest.raises(PermissionError, match="server-created"):
        trusted_tool_context(None)

    runtime = SimpleNamespace(
        state={"user_id": "alice", "tenant_id": "tenant-a", "doc_id": "doc-a"},
        context={"allowed_doc_ids": []},
    )
    context = trusted_tool_context(runtime)
    assert context.allowed_doc_ids == ()
    with pytest.raises(PermissionError):
        context.require_document("doc-a")


@pytest.mark.asyncio
async def test_graphiti_empty_allowlist_does_not_widen_to_user_or_tenant_groups() -> None:
    class _Graphiti:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def search(self, **kwargs: object) -> list[object]:
            self.calls.append(kwargs)
            return []

    graphiti = _Graphiti()
    assert await search_for_risk_context(graphiti, "q", "alice", doc_id="doc-a") == []
    assert await search_for_precedent_chains(graphiti, "q", "alice") == []
    assert await get_obligation_chain(graphiti, "entity", "alice", doc_id="doc-a") == []
    assert graphiti.calls == []


@pytest.mark.asyncio
async def test_graphiti_allowlist_without_tenant_does_not_issue_unscoped_search() -> None:
    class _Graphiti:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def search(self, **kwargs: object) -> list[object]:
            self.calls.append(kwargs)
            return []

    graphiti = _Graphiti()
    assert (
        await search_for_risk_context(
            graphiti,
            "q",
            "alice",
            allowed_doc_ids=("doc-a",),
        )
        == []
    )
    assert (
        await search_for_precedent_chains(
            graphiti,
            "q",
            "alice",
            allowed_doc_ids=("doc-a",),
        )
        == []
    )
    assert (
        await get_obligation_chain(
            graphiti,
            "entity",
            "alice",
            allowed_doc_ids=("doc-a",),
        )
        == []
    )
    assert graphiti.calls == []


def test_evidence_tools_hide_identity_and_document_scope_from_model_schema() -> None:
    from app.features.documents.tools.get_obligation_chain import (
        make_get_obligation_chain_tool,
    )
    from app.features.documents.tools.query_knowledge_graph import (
        make_query_knowledge_graph_tool,
    )
    from app.features.documents.tools.retrieve_statute_section import (
        make_retrieve_statute_section_tool,
    )
    from app.features.documents.tools.search_legal_precedents import (
        make_search_legal_precedents_tool,
    )

    class _Graphiti:
        async def search_for_precedent_chains(self, **_kwargs: Any) -> list[Any]:
            return []

        async def get_obligation_chain(self, **_kwargs: Any) -> list[Any]:
            return []

        async def query_knowledge_graph(self, **_kwargs: Any) -> list[Any]:
            return []

    graphiti = _Graphiti()
    tools = (
        make_search_legal_precedents_tool(graphiti, object(), object()),
        make_retrieve_statute_section_tool(object(), object()),
        make_query_knowledge_graph_tool(graphiti, object()),
        make_get_obligation_chain_tool(graphiti, object()),
    )

    for tool in tools:
        assert "user_id" not in tool.args
        assert "tenant_id" not in tool.args
        assert "allowed_doc_ids" not in tool.args
        assert "runtime" not in tool.args
