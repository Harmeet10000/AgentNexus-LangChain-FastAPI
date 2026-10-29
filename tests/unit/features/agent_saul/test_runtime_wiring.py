"""Scope-agnostic Saul runtime wiring proofs.

Parity evidence: the SAME app.state and the SAME session record drive the
HTTP dependency bundle and the WebSocket handshake through the one runtime
core — scope adapters no longer hold parallel reader logic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import status
from fastapi.exceptions import WebSocketException
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError
from redis.asyncio import Redis
from returns.result import Failure, Success
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.features.agent_saul.dependencies import (
    get_agent_saul_deps,
    get_agent_saul_deps_ws,
    get_agent_saul_ws_security_context,
    get_redis,
    get_redis_ws,
    get_saul_graph,
    get_saul_graph_ws,
)
from app.features.agent_saul.runtime import (
    WS_SESSION_REJECTED_REASON,
    SaulThreadContext,
    WsSessionRejection,
    resolve_ws_session,
    thread_owner_violates,
)
from app.features.auth.adapters.persistence.mongodb.account_repository import SessionData
from app.features.auth.api.websocket.security import (
    WebSocketSecurityContext,
    WebSocketSecurityService,
)
from app.features.auth.domain.errors import AuthInfrastructureError
from app.security.tokens import TokenClaims
from app.utils import ServiceUnavailableException

pytestmark = pytest.mark.unit

_OWNER = "user-owner"
_OTHER = "user-other"


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from returns.result import Result


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _state(**components: object) -> SimpleNamespace:
    return SimpleNamespace(**components)


def _app(state: object) -> SimpleNamespace:
    return SimpleNamespace(state=state)


def _conn(app: object) -> SimpleNamespace:
    """Request/WebSocket duck: the adapters read only ``.app``."""
    return SimpleNamespace(app=app)


def _session(user_id: str, session_id: str = "sess-1") -> SessionData:
    return SessionData(
        session_id=session_id,
        user_id=user_id,
        device_id="device-1",
        created_at=datetime.now(tz=UTC),
        expires_at=datetime.now(tz=UTC) + timedelta(hours=1),
        ttl=3600,
    )


def _claims(user_id: str = _OWNER, session_id: str | None = "sess-1") -> TokenClaims:
    return TokenClaims(
        sub=user_id,
        jti="jti-1",
        sid=session_id,
        role="user",
        permissions=["content:read"],
        token_type="access",
    )


class _StaticSessionStore:
    """Replays one pre-decided session read — the session state under test."""

    def __init__(self, outcome: Result[SessionData | None, Exception]) -> None:
        self._outcome = outcome

    async def get_session(self, session_id: str) -> Result[SessionData | None, Exception]:
        return self._outcome


def _security_service() -> MagicMock:
    security = MagicMock(spec=WebSocketSecurityService)
    security.ensure_connection_capacity = AsyncMock()
    # Bind the REAL context builder so the adapter's return is the auth
    # feature's own WebSocketSecurityContext, not a mock.
    security.build_context = WebSocketSecurityService.build_context
    return security


async def _accept_handshake(
    store: _StaticSessionStore, user_id: str = _OWNER
) -> WebSocketSecurityContext:
    websocket = SimpleNamespace(headers={"origin": "https://legal.example"})
    return await get_agent_saul_ws_security_context(
        websocket,  # type: ignore[arg-type] — request scope duck under test
        _claims(user_id),
        store,
        _security_service(),
    )


# ---------------------------------------------------------------------------
# Runtime bundle: identical construction on both scopes
# ---------------------------------------------------------------------------


async def test_http_and_ws_adapters_yield_the_same_runtime_bundle() -> None:
    graph = MagicMock(spec=CompiledStateGraph)
    checkpointer = MagicMock(spec=AsyncPostgresSaver)
    redis = MagicMock(spec=Redis)
    session_factory = MagicMock(spec=async_sessionmaker)
    app = _app(
        _state(
            saul_graph=graph,
            langgraph_checkpointer=checkpointer,
            redis=redis,
            db_session_local=session_factory,
        )
    )

    http_bundle = await get_agent_saul_deps(_conn(app))  # type: ignore[arg-type]
    ws_bundle = await get_agent_saul_deps_ws(_conn(app))  # type: ignore[arg-type]

    assert http_bundle.graph is ws_bundle.graph is graph
    assert http_bundle.checkpointer is ws_bundle.checkpointer is checkpointer
    assert http_bundle.redis is ws_bundle.redis is redis
    assert http_bundle.session_factory is ws_bundle.session_factory is session_factory


@pytest.mark.parametrize(
    ("http_dep", "ws_dep", "message"),
    [
        (get_saul_graph, get_saul_graph_ws, "Saul graph is not wired"),
        (get_redis, get_redis_ws, "Cache layer is unavailable"),
    ],
)
async def test_unwired_components_fail_identically_in_both_scopes(
    http_dep: Callable[[object], Awaitable[object]],
    ws_dep: Callable[[object], Awaitable[object]],
    message: str,
) -> None:
    app = _app(_state())  # nothing provisioned — including an absent attribute

    with pytest.raises(ServiceUnavailableException, match=message):
        await http_dep(_conn(app))  # type: ignore[arg-type]
    with pytest.raises(ServiceUnavailableException, match=message):
        await ws_dep(_conn(app))  # type: ignore[arg-type]


async def test_wired_redis_component_resolved_from_app_state() -> None:
    redis = MagicMock(spec=Redis)
    app = _app(_state(redis=redis))

    assert await get_redis(_conn(app)) is redis  # type: ignore[arg-type]
    assert await get_redis_ws(_conn(app)) is redis  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Session state parity: shared core and the WS handshake adapter agree
# ---------------------------------------------------------------------------


async def test_session_owner_state_admits_through_core_and_ws_handshake() -> None:
    store = _StaticSessionStore(Success(_session(_OWNER)))

    core = await resolve_ws_session(store, _OWNER, "sess-1")
    context = await _accept_handshake(store)

    assert isinstance(core, Success)
    assert core.unwrap().user_id == _OWNER
    assert isinstance(context, WebSocketSecurityContext)
    assert context.user_id == _OWNER
    assert context.session_id == "sess-1"


async def test_other_user_session_state_refuses_through_both() -> None:
    store = _StaticSessionStore(Success(_session(_OTHER)))  # thread owner mismatch

    core = await resolve_ws_session(store, _OWNER, "sess-1")
    with pytest.raises(WebSocketException) as exc_info:
        await _accept_handshake(store, _OWNER)

    assert isinstance(core, Failure)
    assert core.failure() is WsSessionRejection.OWNER_MISMATCH
    assert exc_info.value.code == status.WS_1008_POLICY_VIOLATION
    assert exc_info.value.reason == WS_SESSION_REJECTED_REASON


async def test_revoked_session_state_refuses_through_core_and_ws_handshake() -> None:
    store = _StaticSessionStore(Success(None))

    core = await resolve_ws_session(store, _OWNER, "sess-1")
    with pytest.raises(WebSocketException) as exc_info:
        await _accept_handshake(store)

    assert isinstance(core, Failure)
    assert core.failure() is WsSessionRejection.REVOKED
    assert exc_info.value.code == status.WS_1008_POLICY_VIOLATION


async def test_redis_outage_fails_closed_through_core_and_ws_handshake() -> None:
    store = _StaticSessionStore(Failure(AuthInfrastructureError(message="redis down")))

    core = await resolve_ws_session(store, _OWNER, "sess-1")
    with pytest.raises(WebSocketException) as exc_info:
        await _accept_handshake(store)

    assert isinstance(core, Failure)
    assert core.failure() is WsSessionRejection.UNAVAILABLE
    assert exc_info.value.code == status.WS_1008_POLICY_VIOLATION


# ---------------------------------------------------------------------------
# Thread context: one construction site, §9 identity gate
# ---------------------------------------------------------------------------


def test_thread_context_correlation_regenerated_per_session() -> None:
    first = SaulThreadContext(user_id=_OWNER, thread_id="thread-1")
    second = SaulThreadContext(user_id=_OWNER, thread_id="thread-1")

    assert first.correlation_id != second.correlation_id


def test_thread_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SaulThreadContext(user_id=_OWNER, thread_id="thread-1", doc_id="doc-1")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("persisted_owner", "caller", "violates"),
    [
        (None, _OWNER, False),  # no persisted owner yet: gate on acquisition alone
        (_OWNER, _OWNER, False),
        (_OTHER, _OWNER, True),
    ],
)
def test_thread_identity_gate_semantics(
    persisted_owner: str | None, caller: str, violates: bool
) -> None:
    assert thread_owner_violates(persisted_owner, caller) is violates
