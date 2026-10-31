"""WebSocket dependency-graph transport safety.

The WS handshake must resolve every dependency through WebSocket-safe
callables: ``fastapi.Request`` and ``HTTPBearer`` cannot execute in a
WebSocket dependency graph, and a shared HTTP-only capability dependency
silently reintroduces them on refactors. These tests pin the invariant at
the route level so the failure surfaces in unit tests, not live handshakes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from fastapi.dependencies.models import Dependant
from fastapi.routing import APIWebSocketRoute
from fastapi.security import HTTPBearer
from returns.result import Success

import app.features.agent_saul.api.websocket.endpoint  # noqa: F401 — registers WS route
from app.features.agent_saul.api.http.router import router
from app.features.auth.api.http.dependencies import AuthorizationContext
from app.features.documents.api.http.dependencies import get_document_authorization_ws
from app.security.tokens import TokenClaims

if TYPE_CHECKING:
    from collections.abc import Iterator


def _ws_routes() -> list[APIWebSocketRoute]:
    return [r for r in router.routes if isinstance(r, APIWebSocketRoute)]


def _walk_dependants(root: Dependant) -> Iterator[Dependant]:
    stack, seen = [root], set()
    while stack:
        dependant = stack.pop()
        if dependant.call in seen:
            continue
        seen.add(dependant.call)
        yield dependant
        stack.extend(dependant.dependencies)


def test_every_ws_dependency_is_transport_safe() -> None:
    routes = _ws_routes()
    assert routes, "agent_saul WS routes must be registered on the feature router"

    offenders: list[str] = []
    for route in routes:
        for dependant in _walk_dependants(route.dependant):
            if dependant.request_param_name is not None:
                offenders.append(f"{route.path}: {dependant.call} requires an HTTP Request")
            if isinstance(dependant.call, HTTPBearer):
                offenders.append(f"{route.path}: HTTPBearer in WS dependency chain")
    assert not offenders, "HTTP-only dependencies leaked into WS routes: " + "; ".join(offenders)


async def test_ws_document_authorization_binds_path_thread_and_tenant_docs() -> None:
    repo = MagicMock()
    repo.list_document_ids_for_tenant = AsyncMock(return_value=Success(("doc-1", "doc-2")))
    claims = TokenClaims(
        sub="user-owner",
        jti="jti-1",
        sid="sess-1",
        role="user",
        permissions=["content:read"],
        token_type="access",
    )
    context = AuthorizationContext(
        claims=claims,
        account=MagicMock(),
        authoritative_tenant_id="tenant-1",
        authoritative_legal_roles=(),
        authoritative_self_review_allowed=False,
        authoritative_role_revision=1,
    )
    websocket = SimpleNamespace(path_params={"thread_id": "thread-9"})

    result = await get_document_authorization_ws(
        websocket,  # type: ignore[arg-type] — only .path_params is read
        context,
        repo,
    )

    assert result.allowed_doc_ids == ("doc-1", "doc-2")
    assert result.thread_id == "thread-9"
    repo.list_document_ids_for_tenant.assert_awaited_once_with(tenant_id="tenant-1")


async def test_ws_document_authorization_falls_back_when_path_thread_missing() -> None:
    repo = MagicMock()
    repo.list_document_ids_for_tenant = AsyncMock(return_value=Success(()))
    claims = TokenClaims(
        sub="user-owner",
        jti="jti-1",
        sid="sess-1",
        role="user",
        permissions=["content:read"],
        token_type="access",
    )
    context = AuthorizationContext(
        claims=claims,
        account=MagicMock(),
        authoritative_tenant_id="tenant-1",
        authoritative_legal_roles=(),
        authoritative_self_review_allowed=False,
        authoritative_role_revision=1,
    )
    websocket = SimpleNamespace(path_params={})

    result = await get_document_authorization_ws(
        websocket,  # type: ignore[arg-type]
        context,
        repo,
    )

    assert result.thread_id == "ws:sess-1"
