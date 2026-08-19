"""
Async PostgreSQL checkpointer for LangGraph persistence.

Uses the existing PostgreSQL connection from app.connections.postgres.

Lifespan wiring:
    from src/app/lifecycle/lifespan.py
    checkpointer = await setup_langgraph_checkpointer(
        conn_string=settings.POSTGRES_URL,
    )
    app.state.langgraph_checkpointer = checkpointer

Dependency injection:
    from src/app/features/agent_saul/dependencies.py
    async def get_saul_checkpointer(request: Request) -> AsyncPostgresSaver:
        return request.app.state.langgraph_checkpointer

"""

from __future__ import annotations

from typing import Any

from app.utils import logger

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except ImportError:
    AsyncPostgresSaver = Any  # ty: ignore[invalid-assignment]


async def setup_langgraph_checkpointer(conn_string: str) -> AsyncPostgresSaver:
    """
    Initialize AsyncPostgresSaver using existing PostgreSQL connection.

    AsyncPostgresSaver creates its own async connection pool,
    reusing the same PostgreSQL database as the app.

    Args:
        conn_string: Async PostgreSQL connection string (postgresql+asyncpg://...)

    Returns:
        Initialized AsyncPostgresSaver ready for use.

    Raises:
        Exception: If connection string is invalid or database is unreachable.
    """

    logger.info("Initializing LangGraph async checkpointer")

    if AsyncPostgresSaver is Any:
        logger.warning("LangGraph Postgres checkpointer is unavailable; skipping initialization")
        return None  # ty: ignore[invalid-return-type]

    try:
        checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)
        await checkpointer.setup()  # ty:ignore[unresolved-attribute]
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.error(
            "Failed to initialize LangGraph checkpointer",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
    else:
        logger.info("LangGraph async checkpointer initialized")
        return checkpointer  # ty:ignore[invalid-return-type]


async def teardown_langgraph_checkpointer(checkpointer: AsyncPostgresSaver | None) -> None:
    """
    Graceful shutdown of AsyncPostgresSaver connection pool.

    Closes all active connections and cleans up resources.

    Args:
        checkpointer: The AsyncPostgresSaver instance to close (can be None).
    """
    if checkpointer is None:
        return

    try:
        if hasattr(checkpointer, "pool") and checkpointer.pool is not None:
            await checkpointer.pool.close()  # ty:ignore[unresolved-attribute]
            logger.info("LangGraph checkpointer connection pool closed")
    except (ConnectionError, OSError) as e:
        logger.warning(
            "Error closing LangGraph checkpointer pool",
            error=str(e),
            error_type=type(e).__name__,
        )
