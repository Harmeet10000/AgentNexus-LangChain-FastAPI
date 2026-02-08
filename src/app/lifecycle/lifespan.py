"""Application lifespan management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import get_settings
from app.connections.mongodb import create_mongo_client
from app.connections.redis import create_redis_client
from app.features.auth.model import User
from app.features.search.model import Search
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    settings = get_settings()

    logger.info("Application starting", app_name=app.title, version=app.version)

    # MongoDB: Initialize using Beanie's recommended approach
    mongo_client, db = await create_mongo_client(
        uri=settings.MONGODB_URI,
        db_name=settings.MONGODB_DB_NAME,
        document_models=[User, Search],
    )
    app.state.mongo_client = mongo_client
    app.state.db = db
    # Redis: Connect and store in app.state
    redis = create_redis_client(settings.REDIS_URL)
    app.state.redis = redis

    # Force early server selection (fail fast if MongoDB unavailable)
    try:
        await mongo_client.admin.command("ping")
        server_info = await mongo_client.server_info()
        logger.info(
            "MongoDB connected",
            database=settings.MONGODB_DB_NAME,
            version=server_info.get("version", "unknown"),
        )
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}", exc_info=True)
        raise

    # Redis: Connect using existing pattern
    try:
        await redis.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}", exc_info=True)
        # Don't raise - Redis is optional for some features

    logger.info("Application ready", status="running")

    yield

    # --- Shutdown ---

    logger.info("Application shutting down", status="stopping")

    if hasattr(app.state, "mongo_client"):
        app.state.mongo_client.close()
        logger.info("MongoDB connection closed")

    if hasattr(app.state, "redis"):
        await app.state.redis.close()
        logger.info("Redis connection closed")
    logger.info("Application shutdown complete", status="stopped")
