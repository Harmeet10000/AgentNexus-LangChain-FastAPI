"""Application lifespan management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.connections.mongodb import close_mongodb_connection, connect_to_mongodb
from app.connections.postgres import close_db, init_db
from app.connections.redis import close_redis_connection, connect_to_redis
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    # setup_logging()
    logger.info("Application starting", app_name=app.title, version=app.version)

    # Connect to databases
    await connect_to_mongodb()
    await connect_to_redis()
    await init_db()

    logger.info("Application ready", status="running")

    yield

    logger.info("Application shutting down", status="stopping")

    # Close database connections
    await close_mongodb_connection()
    await close_redis_connection()
    await close_db()

    logger.info("Application shutdown complete", status="stopped")
