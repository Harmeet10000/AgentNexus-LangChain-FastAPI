from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest

# The root tests/conftest.py stubs heavy/cyclic modules with MagicMock so unit
# tests stay import-light. The real app needs the real modules, so drop the
# stubs before building it. Kept in one place so Phase 6 can retire them.
_STUBBED_MODULES = [
    "app.connections.mcp",
    "app.connections.celery",
    "mcp_core",
    "mcp_core.client.auth",
    "mcp_core.client.manager",
    "mcp_core.client.settings",
    "mcp_core.common.errors",
    "mcp_core.common.models",
    "mcp_core.lifespan_mcp",
    "mcp_core.mcp",
    "mcp_core.server.factory",
    "mcp_core.server.http",
    "mcp_core.server.middleware",
    "mcp_core.server.tools",
    "tasks",
    "tasks.auth_email_tasks",
    "tasks.search_tasks",
    "app.shared.langgraph_layer",
    "app.shared.langgraph_layer.agent_saul.state",
    "app.shared.langgraph_layer.checkpointer",
    "app.shared.langgraph_layer.kb_retry",
    "app.shared.langgraph_layer.retrieval_kb",
    "app.features.auth.token_audit_log",
]


@pytest.fixture
def client() -> Iterator[Any]:
    """Real-stack app client with lifespan running.

    Requires a live PostgreSQL, Redis, and MongoDB. Gated by the
    ``integration`` marker; excluded from the default ``pytest`` run.
    """
    for name in _STUBBED_MODULES:
        sys.modules.pop(name, None)

    from app.main import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client