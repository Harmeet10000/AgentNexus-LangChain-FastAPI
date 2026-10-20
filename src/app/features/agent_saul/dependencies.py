"""
FastAPI dependencies for agent_saul.

All infra clients are read from request.app.state — the single source of
truth set during lifespan.  No globals.

Lifespan callers must set:
    app.state.saul_graph            → CompiledStateGraph (built by factory.build_saul_graph)
    app.state.langgraph_checkpointer → AsyncPostgresSaver
    app.state.redis                 → redis.asyncio.Redis
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, Request, WebSocket, WebSocketException, status
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from returns.result import Failure
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.features.agent_saul.transcript.export import TranscriptExporter
from app.features.agent_saul.transcript.repository import ChatTranscriptRepository
from app.features.auth.dependencies import WebSocketClaims, get_refresh_token_repository
from app.features.auth.repository import RefreshTokenRepository
from app.features.auth.websocket_security import WebSocketSecurityContext, WebSocketSecurityService
from app.utils import ServiceUnavailableException

# ---------------------------------------------------------------------------
# Individual dependency extractors
# ---------------------------------------------------------------------------


async def get_saul_graph(request: Request) -> CompiledStateGraph[Any]:
    # getattr, not attribute access: Starlette's State *raises* on an unknown
    # attribute, so a bare read turns "never provisioned" into an unhandled
    # AttributeError and a 500 instead of the typed 503 below.
    graph = getattr(request.app.state, "saul_graph", None)
    if graph is None:
        msg = "Saul graph is not wired"
        raise ServiceUnavailableException(msg)
    return graph


async def get_saul_graph_ws(websocket: WebSocket) -> CompiledStateGraph[Any]:
    """WS-scope sibling of `get_saul_graph`.

    FastAPI cannot inject `Request` inside a websocket scope: the shared
    state readers below must read the SAME app.state through `WebSocket`
    or every Saul WS handshake dies with
    ``get_saul_graph() missing 1 required positional argument 'request'``.
    """
    graph = getattr(websocket.app.state, "saul_graph", None)
    if graph is None:
        msg = "Saul graph is not wired"
        raise ServiceUnavailableException(msg)
    return graph


async def get_saul_checkpointer(request: Request) -> AsyncPostgresSaver:
    checkpointer = getattr(request.app.state, "langgraph_checkpointer", None)
    if checkpointer is None:
        message = "Persistence layer is unavailable"
        raise ServiceUnavailableException(message)
    return checkpointer


async def get_saul_checkpointer_ws(websocket: WebSocket) -> AsyncPostgresSaver:
    """WS-scope sibling of `get_saul_checkpointer` (same `Request` limitation)."""
    checkpointer = getattr(websocket.app.state, "langgraph_checkpointer", None)
    if checkpointer is None:
        message = "Persistence layer is unavailable"
        raise ServiceUnavailableException(message)
    return checkpointer


async def get_redis(request: Request) -> Redis:
    redis = request.app.state.redis
    if redis is None:
        message = "Cache layer is unavailable"
        raise ServiceUnavailableException(message)
    return redis


async def get_redis_ws(websocket: WebSocket) -> Redis:
    """WS-scope sibling of `get_redis` (same `Request` limitation)."""
    redis = websocket.app.state.redis
    if redis is None:
        message = "Cache layer is unavailable"
        raise ServiceUnavailableException(message)
    return redis


async def get_db_session_local(
    request: Request,
) -> async_sessionmaker[AsyncSession]:
    """Sync-side session maker for out-of-request-scope work (the transcript
    projector runs at WS boundaries, not inside a request dependency)."""
    session_local = getattr(request.app.state, "db_session_local", None)
    if session_local is None:
        message = "Database layer is unavailable"
        raise ServiceUnavailableException(message)
    return session_local


async def get_db_session_local_ws(
    websocket: WebSocket,
) -> async_sessionmaker[AsyncSession]:
    """WS-scope sibling of `get_db_session_local` (same `Request` limitation)."""
    session_local = getattr(websocket.app.state, "db_session_local", None)
    if session_local is None:
        message = "Database layer is unavailable"
        raise ServiceUnavailableException(message)
    return session_local


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
# Bundled context object — avoids long parameter lists at orchestration layer
# ---------------------------------------------------------------------------


class AgentSaulDeps(BaseModel):
    """Narrow context object for Agent Saul dependencies.

    Typed against infra protocols so nodes remain decoupled from concrete
    client implementations in tests.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: CompiledStateGraph[Any]
    checkpointer: AsyncPostgresSaver
    redis: Redis
    session_factory: async_sessionmaker[AsyncSession]


async def get_agent_saul_deps(
    graph: Annotated[CompiledStateGraph[Any], Depends(get_saul_graph)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_saul_checkpointer)],
    redis: Annotated[Redis, Depends(get_redis)],
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_db_session_local)],
) -> AgentSaulDeps:
    return AgentSaulDeps(
        graph=graph, checkpointer=checkpointer, redis=redis, session_factory=session_factory
    )


async def get_agent_saul_deps_ws(
    graph: Annotated[CompiledStateGraph[Any], Depends(get_saul_graph_ws)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_saul_checkpointer_ws)],
    redis: Annotated[Redis, Depends(get_redis_ws)],
    session_factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_db_session_local_ws)],
) -> AgentSaulDeps:
    """WS-scope composition of the same bundle — identical shape, WS readers."""
    return AgentSaulDeps(
        graph=graph, checkpointer=checkpointer, redis=redis, session_factory=session_factory
    )


async def get_agent_saul_ws_security_context(
    websocket: WebSocket,
    claims: WebSocketClaims,
    token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
    security_service: Annotated[WebSocketSecurityService, Depends(get_websocket_security_service)],
) -> WebSocketSecurityContext:
    origin = websocket.headers.get("origin")
    security_service.ensure_origin_allowed(origin)

    if claims.sid is not None:
        session_result = await token_repo.get_session(claims.sid)
        if isinstance(session_result, Failure):
            # Fail closed: a Redis outage must not authenticate the handshake.
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Session expired or revoked",
            )
        session = session_result.unwrap()
        if session is None or session.user_id != claims.sub:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Session expired or revoked",
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
