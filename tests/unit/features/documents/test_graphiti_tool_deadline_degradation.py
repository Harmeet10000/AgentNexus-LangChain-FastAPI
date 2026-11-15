"""Graphiti knowledge tools count a deadline miss as the outage it is (harden 5.2).

A deadline miss used to answer ``unavailable_result`` without touching the
degradation counter: the standing alert could only ever see the
BoundGraphitiService-degraded reads, so every slow-Graphiti miss was invisible.
The miss path now records the same ``GRAPHITI_UNAVAILABLE`` degradation the
transport-failure path does.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from langchain.tools import ToolRuntime

from app.config.settings import get_settings
from app.resilience.idempotency import IdempotencyGuard

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

_DOC = "doc-a"


def _null_idempotency() -> IdempotencyGuard:
    redis = MagicMock()

    async def _none(*_args: Any, **_kwargs: Any) -> None:
        return None

    redis.get = _none
    redis.set = _none

    class _Unreachable:
        def connect(self) -> None:
            message = "no database in unit test"
            raise RuntimeError(message)

    return IdempotencyGuard(redis=redis, db_engine=_Unreachable())


def _runtime() -> ToolRuntime:
    """Server-created runtime granting document.read over one document."""
    return ToolRuntime(
        state={},
        context={
            "user_id": "alice",
            "tenant_id": "personal:alice",
            "allowed_doc_ids": (_DOC,),
            "allowed_operations": ("document.read",),
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


class _HangingGraphitiService:
    """Every read hangs: the Graphiti deadline is the only thing that fires."""

    def __getattr__(self, name: str) -> Callable[..., Any]:
        async def _hang(**_kwargs: Any) -> list[Any]:
            await asyncio.sleep(30)
            return []

        return _hang


@pytest.fixture
def _short_graphiti_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENTS_RETRIEVAL_GRAPHITI_TIMEOUT_SECONDS", "0.05")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_short_graphiti_deadline")
async def test_query_knowledge_graph_deadline_miss_counts_the_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.documents.application.tools import query_knowledge_graph as tool_module
    from app.features.documents.application.tools.query_knowledge_graph import (
        make_query_knowledge_graph_tool,
    )

    recorded: list[SimpleNamespace] = []
    monkeypatch.setattr(
        tool_module,
        "record_degradation",
        lambda kind, **kwargs: recorded.append(SimpleNamespace(kind=kind, **kwargs)),
    )
    tool = make_query_knowledge_graph_tool(_HangingGraphitiService(), _null_idempotency())
    result = await tool.ainvoke({"query": "termination risk", "runtime": _runtime()})

    assert result["success"] is False
    assert "deadline" in str(result)
    assert [
        event
        for event in recorded
        if event.kind == tool_module.GRAPHITI_UNAVAILABLE
        and event.surface == "query_knowledge_graph"
    ], "a deadline miss must feed the standing degradation alert"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_short_graphiti_deadline")
async def test_get_obligation_chain_deadline_miss_counts_the_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.documents.application.tools import get_obligation_chain as tool_module
    from app.features.documents.application.tools.get_obligation_chain import (
        make_get_obligation_chain_tool,
    )

    recorded: list[SimpleNamespace] = []
    monkeypatch.setattr(
        tool_module,
        "record_degradation",
        lambda kind, **kwargs: recorded.append(SimpleNamespace(kind=kind, **kwargs)),
    )
    tool = make_get_obligation_chain_tool(_HangingGraphitiService(), _null_idempotency())
    result = await tool.ainvoke({"entity_name": "Acme Corp", "runtime": _runtime()})

    assert result["success"] is False
    assert "deadline" in str(result)
    assert [
        event
        for event in recorded
        if event.kind == tool_module.GRAPHITI_UNAVAILABLE
        and event.surface == "get_obligation_chain"
    ], "a deadline miss must feed the standing degradation alert"
