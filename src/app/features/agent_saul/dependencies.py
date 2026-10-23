"""
FastAPI adapters for agent_saul.

All infra clients are read from request.app.state — the single source of
truth set during lifespan.  No globals.

The scope-agnostic implementation (state component reads, the runtime
bundle, the fail-closed handshake session read, thread context) lives in
runtime.py. This module adds only the transport plumbing: FastAPI cannot
inject `Request` inside a websocket scope (and `WebSocket` inside an HTTP
scope), so each scope exposes a one-line adapter over the same core, and
every WS handshake keeps exactly the HTTP chain's auth semantics.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, Request, WebSocket, WebSocketException, status
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis
from returns.result import Failure
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.features.agent_saul.runtime import (
    WS_SESSION_REJECTED_REASON,
    AgentSaulDeps,
    SessionStore,
    WsSessionRejection,
    build_agent_saul_runtime,
    require_db_session_local,
    require_saul_checkpointer,
    require_saul_graph,
    require_saul_redis,
    resolve_ws_session,
)
from app.features.agent_saul.transcript.export import TranscriptExporter
from app.features.agent_saul.transcript.repository import ChatTranscriptRepository
from app.features.auth.dependencies import WebSocketClaims, get_refresh_token_repository
from app.features.auth.websocket_security import WebSocketSecurityContext, WebSocketSecurityService
from app.observability.logging import logger

# The `_ws` variants below are WS-scoped siblings of their HTTP twins, not
# parallel implementations: each body is one call into runtime.py, so a guard
# can no longer drift on one scope only (the 7aeba9f / 7f04409 split).


# ---------------------------------------------------------------------------
# Scope adapters — HTTP (Request) and WS (WebSocket) one-liners
# ---------------------------------------------------------------------------


async def get_saul_graph(request: Request) -> CompiledStateGraph[Any]:
    return require_saul_graph(request.app.state)


async def get_saul_graph_ws(websocket: WebSocket) -> CompiledStateGraph[Any]:
    return require_saul_graph(websocket.app.state)


async def get_saul_checkpointer(request: Request) -> AsyncPostgresSaver:
    return require_saul_checkpointer(request.app.state)


async def get_saul_checkpointer_ws(websocket: WebSocket) -> AsyncPostgresSaver:
    return require_saul_checkpointer(websocket.app.state)


async def get_redis(request: Request) -> Redis:
    return require_saul_redis(request.app.state)


async def get_redis_ws(websocket: WebSocket) -> Redis:
    return require_saul_redis(websocket.app.state)


async def get_db_session_local(
    request: Request,
) -> async_sessionmaker[AsyncSession]:
    """Sync-side session maker for out-of-request-scope work (the transcript
    projector runs at WS boundaries, not inside a request dependency)."""
    return require_db_session_local(request.app.state)


async def get_db_session_local_ws(
    websocket: WebSocket,
) -> async_sessionmaker[AsyncSession]:
    return require_db_session_local(websocket.app.state)


async def get_websocket_security_service(websocket: WebSocket) -> WebSocketSecurityService:
    return websocket.app.state.websocket_security


async def get_transcript_repository(
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_db_session_local)],
) -> ChatTranscriptRepository:
    """The transcript projector/repository singleton per request scope."""
    return ChatTranscriptRepository(session_factory)


async def get_transcript_exporter(
    repository: Annotated[ChatTranscriptRepository, Depends(get_transcript_repository)],
) -> TranscriptExporter:
    """Deep transcript export operation — owns pagination, cursors, schema gating."""
    return TranscriptExporter(repository)


# ---------------------------------------------------------------------------
# Runtime bundle — one runtime bundle for both scopes
# ---------------------------------------------------------------------------


async def get_agent_saul_deps(request: Request) -> AgentSaulDeps:
    return build_agent_saul_runtime(request.app.state)


async def get_agent_saul_deps_ws(websocket: WebSocket) -> AgentSaulDeps:
    return build_agent_saul_runtime(websocket.app.state)


async def get_agent_saul_ws_security_context(
    websocket: WebSocket,
    claims: WebSocketClaims,
    token_repo: Annotated[SessionStore, Depends(get_refresh_token_repository)],
    security_service: Annotated[WebSocketSecurityService, Depends(get_websocket_security_service)],
) -> WebSocketSecurityContext:
    """WebSocket handshake adapter over runtime's fail-closed session core.

    Same admission semantics as the HTTP chain: session absent/revoked or
    owned by another user (or an unreachable Redis) → WS_1008, never open.
    """
    origin = websocket.headers.get("origin")
    security_service.ensure_origin_allowed(origin)

    if claims.sid is not None:
        session_result = await resolve_ws_session(token_repo, claims.sub, claims.sid)
        if isinstance(session_result, Failure):
            rejection = session_result.failure()
            if rejection is WsSessionRejection.UNAVAILABLE:
                # Fail closed: a Redis outage must not authenticate the handshake.
                logger.bind(session_id=claims.sid, user_id=claims.sub).warning(
                    "saul_ws_session_check_unavailable"
                )
            # Absent and owner-mismatched rejections ride the same wire reason;
            # the enum distinguishes them for observability only.
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason=WS_SESSION_REJECTED_REASON,
            )

    await security_service.ensure_connection_capacity(claims.sub)
    return security_service.build_context(
        claims=claims,
        origin=origin,
        connection_id=str(uuid4()),
    )


# ---------------------------------------------------------------------------
# Annotated aliases — reused across router handlers
# ---------------------------------------------------------------------------

AgentSaulDepsAnnotated = Annotated[AgentSaulDeps, Depends(get_agent_saul_deps)]
AgentSaulDepsWsAnnotated = Annotated[AgentSaulDeps, Depends(get_agent_saul_deps_ws)]
AgentSaulWebSocketSecurityContextAnnotated = Annotated[
    WebSocketSecurityContext,
    Depends(get_agent_saul_ws_security_context),
]
TranscriptExporterDep = Annotated[TranscriptExporter, Depends(get_transcript_exporter)]
