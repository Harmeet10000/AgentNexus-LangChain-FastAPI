"""
FastAPI dependencies for agent_saul.

All infra clients are read from request.app.state — the single source of
truth set during lifespan.  No globals.

Lifespan callers must set:
    app.state.saul_graph        → CompiledStateGraph (built by factory.build_saul_graph)
    app.state.saul_checkpointer → AsyncPostgresSaver
    app.state.redis             → redis.asyncio.Redis
"""

from dataclasses import dataclass
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, Request, WebSocket, WebSocketException, status
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.features.auth.dependencies import (
    WebSocketClaims,
    get_refresh_token_repository,
)
from app.features.auth.repository import RefreshTokenRepository
from app.features.auth.websocket_security import (
    WebSocketSecurityContext,
    WebSocketSecurityService,
)

# ---------------------------------------------------------------------------
# Individual dependency extractors
# ---------------------------------------------------------------------------


async def get_saul_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.saul_graph


async def get_saul_checkpointer(request: Request) -> AsyncPostgresSaver:
    return request.app.state.saul_checkpointer


async def get_redis(request: Request) -> Redis:
    return request.app.state.redis


async def get_current_user_id(request: Request) -> str:
    """
    Stub — replace with your project's JWT/session auth dependency.
    The user_id is injected into LegalAgentState.user_id and
    used as the LangGraph Store namespace for long-term memory.
    """
    # Example: return request.state.user_id after auth middleware sets it.
    return request.state.user_id


async def get_websocket_security_service(websocket: WebSocket) -> WebSocketSecurityService:
    return websocket.app.state.websocket_security


# ---------------------------------------------------------------------------
# Bundled context object — avoids long parameter lists at orchestration layer
# ---------------------------------------------------------------------------


@dataclass
class AgentSaulDeps:
    """Narrow context object for Agent Saul dependencies.

    Typed against infra protocols so nodes remain decoupled from concrete
    client implementations in tests.
    """

    graph: CompiledStateGraph
    checkpointer: AsyncPostgresSaver
    redis: Redis


async def get_agent_saul_deps(
    graph: Annotated[CompiledStateGraph, Depends(get_saul_graph)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_saul_checkpointer)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> AgentSaulDeps:
    return AgentSaulDeps(graph=graph, checkpointer=checkpointer, redis=redis)


async def get_agent_saul_ws_security_context(
    websocket: WebSocket,
    claims: WebSocketClaims,
    token_repo: Annotated[RefreshTokenRepository, Depends(get_refresh_token_repository)],
    security_service: Annotated[WebSocketSecurityService, Depends(get_websocket_security_service)],
) -> WebSocketSecurityContext:
    origin = websocket.headers.get("origin")
    security_service.ensure_origin_allowed(origin)

    if claims.sid is not None:
        session = await token_repo.get_session(claims.sid)
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
CurrentUserIdAnnotated = Annotated[str, Depends(get_current_user_id)]
AgentSaulWebSocketSecurityContextAnnotated = Annotated[
    WebSocketSecurityContext,
    Depends(get_agent_saul_ws_security_context),
]
