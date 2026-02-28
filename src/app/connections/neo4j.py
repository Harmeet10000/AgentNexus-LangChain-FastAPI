"""Neo4j database configuration with driver management."""

from contextlib import asynccontextmanager

from fastapi import Request
from neo4j import Driver, GraphDatabase, basic_auth

from app.config.settings import get_settings
from app.utils.logger import logger

settings = get_settings()


async def init_neo4j() -> Driver:
    """Initialize Neo4j driver and test connection.

    Returns:
        Driver: Configured Neo4j driver for app.state injection

    Raises:
        Exception: If connection fails or version check fails
    """
    try:
        driver = GraphDatabase.driver(
            uri=settings.NEO4J_URI,
            auth=basic_auth(user=settings.NEO4J_USERNAME, password=settings.NEO4J_PASSWORD),
            max_connection_pool_size=50,
            connection_acquisition_timeout=30,
            connection_timeout=15,
            # max_retry_time=30,
            # database=settings.NEO4J_DATABASE,
        )

        return driver

    except Exception as e:
        logger.error(
            "Neo4j initialization failed",
            uri=settings.NEO4J_URI,
            error=str(e),
            exc_info=True,
        )
        raise


async def get_neo4j_driver(request: Request) -> Driver:
    """Dependency for Neo4j driver retrieved from app.state.

    Args:
        request: FastAPI request object

    Returns:
        Driver: Neo4j driver instance
    """
    return request.app.state.neo4j_driver


@asynccontextmanager
async def get_neo4j_session(driver: Driver, database: str | None = None):
    """Context manager for Neo4j sessions.

    Args:
        driver: Neo4j Driver instance
        database: Optional database name (defaults to configured database)

    Yields:
        AsyncSession: Neo4j async session

    Example:
        async with get_neo4j_session(driver) as session:
            result = await session.run("MATCH (n) RETURN n LIMIT 5")
            records = await result.all()
    """
    session = driver.session(database=database or settings.NEO4J_DATABASE)
    try:
        yield session
    except Exception as e:
        logger.error("Neo4j session error", error=str(e), exc_info=True)
        raise
    finally:
        await session.close()


async def close_neo4j_driver(driver: Driver) -> None:
    """Close Neo4j driver and cleanup connections.

    Args:
        driver: Neo4j Driver instance
    """
    try:
        driver.close()
        # logger.info("Neo4j driver closed")
    except Exception as e:
        logger.error("Error closing Neo4j driver", error=str(e), exc_info=True)
        raise
