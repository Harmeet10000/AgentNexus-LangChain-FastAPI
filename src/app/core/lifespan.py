"""Application lifespan management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.connections.mongodb import connect_to_mongodb
from app.connections.postgres import init_db
from app.connections.redis import connect_to_redis
from app.utils.logger import logger
from app.utils.uvicorn_logger import setup_uvicorn_logging


@asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    # setup_logging()
    setup_uvicorn_logging()
    logger.info("Application starting")

    # Connect to databases
    await connect_to_mongodb()
    await connect_to_redis()
    await init_db()

    yield

    logger.info("Application shutdown")
