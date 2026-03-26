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

from fastapi import Depends, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

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


# ---------------------------------------------------------------------------
# Annotated aliases — reused across router handlers
# ---------------------------------------------------------------------------

AgentSaulDepsAnnotated = Annotated[AgentSaulDeps, Depends(get_agent_saul_deps)]
CurrentUserIdAnnotated = Annotated[str, Depends(get_current_user_id)]
