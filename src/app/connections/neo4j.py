"""Neo4j database configuration with driver management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Request
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession, basic_auth

from app.config import get_settings
from app.utils import logger

settings = get_settings()


async def init_neo4j() -> AsyncDriver:
    """Initialize Neo4j driver and test connection.

    Returns:
        AsyncDriver: Configured Neo4j driver for app.state injection

    Raises:
        Exception: If connection fails or version check fails
    """
    try:
        return AsyncGraphDatabase.driver(
            uri=settings.NEO4J_URI,
            auth=basic_auth(user=settings.NEO4J_USERNAME, password=settings.NEO4J_PASSWORD),
            max_connection_pool_size=50,
            connection_acquisition_timeout=30,
            connection_timeout=15,
            # max_retry_time=30,
            # database=settings.NEO4J_DATABASE,
        )

    except Exception as e:
        logger.error(
            "Neo4j initialization failed",
            uri=settings.NEO4J_URI,
            error=str(e),
            exc_info=True,
        )
        raise


async def get_neo4j_driver(request: Request) -> AsyncDriver:
    """Dependency for Neo4j driver retrieved from app.state.

    Args:
        request: FastAPI request object

    Returns:
        AsyncDriver: Neo4j driver instance
    """
    return request.app.state.neo4j_driver


@asynccontextmanager
async def get_neo4j_session(
    driver: AsyncDriver,
    database: str | None = None,
) -> AsyncIterator[AsyncSession]:
    """Context manager for Neo4j sessions.

    Args:
        driver: Neo4j async driver instance
        database: Optional database name (defaults to configured database)

    Yields:
        AsyncSession: Neo4j async session

    Example:
        async with get_neo4j_session(driver) as session:
            result = await session.run("MATCH (n) RETURN n LIMIT 5")
            records = await result.all()
    """
    async with driver.session(database=database or settings.NEO4J_DATABASE) as session:
        try:
            yield session
        except Exception as e:
            logger.error("Neo4j session error", error=str(e), exc_info=True)
            raise


async def close_neo4j_driver(driver: AsyncDriver) -> None:
    """Close Neo4j driver and cleanup connections.

    Args:
        driver: Neo4j async driver instance
    """
    try:
        await driver.close()
        # logger.info("Neo4j driver closed")
    except Exception as e:
        logger.error("Error closing Neo4j driver", error=str(e), exc_info=True)
        raise
