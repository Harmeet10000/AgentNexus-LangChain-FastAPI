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

from typing import TYPE_CHECKING, Annotated
from uuid import uuid4

from fastapi import Depends, Request, WebSocket, WebSocketException, status
from pydantic import BaseModel, ConfigDict

from app.features.auth import WebSocketSecurityContext
from app.features.auth.dependencies import get_refresh_token_repository
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from typing import Any

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph.state import CompiledStateGraph
    from redis.asyncio import Redis

    from app.features.auth import RefreshTokenRepository, WebSocketSecurityService
    from app.features.auth.dependencies import WebSocketClaims

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


async def get_saul_checkpointer(request: Request) -> AsyncPostgresSaver:
    checkpointer = getattr(request.app.state, "langgraph_checkpointer", None)
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


async def get_websocket_security_service(websocket: WebSocket) -> WebSocketSecurityService:
    return websocket.app.state.websocket_security


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


async def get_agent_saul_deps(
    graph: Annotated[CompiledStateGraph[Any], Depends(get_saul_graph)],
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
AgentSaulWebSocketSecurityContextAnnotated = Annotated[
    WebSocketSecurityContext,
    Depends(get_agent_saul_ws_security_context),
]
