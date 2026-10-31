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
from unittest.mock import AsyncMock

import pytest
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIWebSocketRoute
from fastapi.security import HTTPBearer
from returns.result import Failure, Success

import app.features.agent_saul.api.websocket.endpoint  # noqa: F401 — registers WS route
from app.features.agent_saul.api.http.router import router
from app.features.auth.api.http.dependencies import AuthorizationContext
from app.features.documents.api.http import dependencies as document_dependencies
from app.features.documents.api.http.dependencies import get_document_authorization_ws
from app.security.tokens import TokenClaims
from app.utils import InfrastructureException

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Session:
    """Minimal async-session double that records lifecycle calls."""

    def __init__(self) -> None:
        self.closed = False
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        self.closed = True
        return False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class _SessionFactory:
    """Callable factory returning one instrumented session."""

    def __init__(self, session: _Session) -> None:
        self.session = session
        self.calls = 0

    def __call__(self) -> _Session:
        self.calls += 1
        return self.session


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


def _authorization_context() -> AuthorizationContext:
    claims = TokenClaims(
        sub="user-owner",
        jti="jti-1",
        sid="sess-1",
        role="user",
        permissions=["content:read"],
        token_type="access",
    )
    return AuthorizationContext(
        claims=claims,
        account=SimpleNamespace(id="user-owner"),
        authoritative_tenant_id="tenant-1",
        authoritative_legal_roles=(),
        authoritative_self_review_allowed=False,
        authoritative_role_revision=1,
    )


def _patch_repository(monkeypatch: pytest.MonkeyPatch, result: object) -> SimpleNamespace:
    repo = SimpleNamespace(list_document_ids_for_tenant=AsyncMock(return_value=result))
    monkeypatch.setattr(document_dependencies, "DocumentRepository", lambda _session: repo)
    return repo


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


def test_saul_ws_route_depends_on_ws_document_authorization() -> None:
    route = next(r for r in _ws_routes() if r.path.endswith("/ws/{thread_id}"))
    calls = {dependant.call for dependant in _walk_dependants(route.dependant)}
    assert get_document_authorization_ws in calls


async def test_ws_document_authorization_binds_path_thread_and_tenant_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _patch_repository(monkeypatch, Success(("doc-1", "doc-2")))
    session = _Session()
    session_factory = _SessionFactory(session)
    websocket = SimpleNamespace(path_params={"thread_id": "thread-9"})

    result = await get_document_authorization_ws(
        websocket,
        _authorization_context(),
        session_factory,
    )

    assert result.allowed_doc_ids == ("doc-1", "doc-2")
    assert result.thread_id == "thread-9"
    repo.list_document_ids_for_tenant.assert_awaited_once_with(tenant_id="tenant-1")
    assert session_factory.calls == 1
    assert session.closed is True
    assert session.committed is True


async def test_ws_document_authorization_falls_back_when_path_thread_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_repository(monkeypatch, Success(()))
    session = _Session()
    session_factory = _SessionFactory(session)
    websocket = SimpleNamespace(path_params={})

    result = await get_document_authorization_ws(
        websocket,
        _authorization_context(),
        session_factory,
    )

    assert result.thread_id == "ws:sess-1"
    assert session.closed is True


async def test_ws_document_authorization_closes_session_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_repository(monkeypatch, Failure("repository unavailable"))
    session = _Session()
    session_factory = _SessionFactory(session)
    websocket = SimpleNamespace(path_params={"thread_id": "thread-9"})

    with pytest.raises(InfrastructureException, match="tenant document capability"):
        await get_document_authorization_ws(
            websocket,
            _authorization_context(),
            session_factory,
        )

    assert session.closed is True
    assert session.rolled_back is True
