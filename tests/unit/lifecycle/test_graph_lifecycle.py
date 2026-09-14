"""Process-scoped graph provider and API provisioning proofs."""

from __future__ import annotations

import importlib
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.features.agent_saul.dependencies import get_agent_saul_deps
from app.lifecycle import graphs

lifespan_module = importlib.import_module("app.lifecycle.lifespan")

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

pytestmark = pytest.mark.unit


def test_importing_graph_providers_compiles_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls: list[object] = []
    original_compile = StateGraph.compile

    def recording_compile(self: StateGraph[Any], *args: object, **kwargs: object) -> object:
        compile_calls.append(self)
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(StateGraph, "compile", recording_compile)
    sys.modules.pop("app.lifecycle.graphs", None)

    importlib.import_module("app.lifecycle.graphs")

    assert compile_calls == []


def test_memory_provider_uses_the_configured_partition_prefix() -> None:
    settings = cast("Any", SimpleNamespace(COGNEE_DATASET_PREFIX="tenant-memory"))

    service = graphs.provide_agent_memory_service(settings)

    assert service._prefix == "tenant-memory"


async def test_failed_graph_policy_degrades_and_leaves_capability_absent() -> None:
    app = FastAPI()

    async def fail(_app: FastAPI, _settings: object) -> None:
        message = "compile failed"
        raise RuntimeError(message)

    policy = lifespan_module.StartupPolicy(
        name="test_graph",
        setup=fail,
        state_attr="test_graph",
        fatal_on=(),
        degrade_on=(Exception,),
        report=lambda _exc: None,
    )

    await lifespan_module._run_startup_policy(app, object(), policy)

    assert app.state.test_graph is None


def test_open_deep_search_import_compiles_no_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls: list[object] = []
    original_compile = StateGraph.compile

    def recording_compile(self: StateGraph[Any], *args: object, **kwargs: object) -> object:
        compile_calls.append(self)
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(StateGraph, "compile", recording_compile)
    sys.modules.pop("app.shared.langgraph_layer.open_deep_search.graph", None)

    module = importlib.import_module("app.shared.langgraph_layer.open_deep_search.graph")

    assert compile_calls == []
    assert module.supervisor_subgraph is None
    assert module.researcher_subgraph is None
    assert module.deep_researcher is None


def test_started_application_exposes_agent_saul_dependency_bundle() -> None:
    graph = MagicMock(spec=CompiledStateGraph)
    checkpointer = MagicMock(spec=AsyncPostgresSaver)
    redis = MagicMock(spec=Redis)

    @asynccontextmanager
    async def started(app: FastAPI) -> AsyncIterator[None]:
        app.state.saul_graph = graph
        app.state.langgraph_checkpointer = checkpointer
        app.state.redis = redis
        yield

    app = FastAPI(lifespan=started)

    @app.get("/saul-ready")
    async def saul_ready(
        deps: Any = Depends(get_agent_saul_deps),  # noqa: FAST002 - test-only probe
    ) -> dict[str, bool]:
        return {
            "graph": deps.graph is graph,
            "checkpointer": deps.checkpointer is checkpointer,
            "redis": deps.redis is redis,
        }

    with TestClient(app) as client:
        response = client.get("/saul-ready")

    assert response.status_code == 200
    assert response.json() == {"graph": True, "checkpointer": True, "redis": True}
