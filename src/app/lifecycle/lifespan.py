"""Application lifespan management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import get_settings
from app.connections.mongodb import create_mongo_client
from app.connections.postgres import init_db
from app.connections.redis import create_redis_client
from app.features.auth.model import User
from app.features.search.model import Search
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    settings = get_settings()

    logger.info("Application starting", app_name=app.title, version=app.version)

    # PostgreSQL: Initialize engine and session factory
    try:
        db_engine, db_session_local = await init_db()
        app.state.db_engine = db_engine
        app.state.db_session_local = db_session_local
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}", exc_info=True)
        raise

    # MongoDB: Initialize using Beanie's recommended approach
    try:
        mongo_client, db = await create_mongo_client(
            uri=settings.MONGODB_URI,
            db_name=settings.MONGODB_DB_NAME,
            document_models=[User, Search],
        )
        app.state.mongo_client = mongo_client
        app.state.db = db
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

    # Redis: Connect and store in app.state
    try:
        redis = create_redis_client(settings.REDIS_URL)
        await redis.ping()
        app.state.redis = redis
        logger.info("Redis connected")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}", exc_info=True)

    logger.info("Application ready", status="running")

    yield

    # --- Shutdown (cleanup in reverse order) ---

    logger.info("Application shutting down", status="stopping")

    if hasattr(app.state, "redis"):
        try:
            await app.state.redis.close()
            logger.info("Redis connection closed")
        except Exception:
            logger.error("Redis close failed", exc_info=True)

    if hasattr(app.state, "mongo_client"):
        try:
            app.state.mongo_client.close()
            logger.info("MongoDB connection closed")
        except Exception:
            logger.error("MongoDB close failed", exc_info=True)

    if hasattr(app.state, "db_engine"):
        try:
            await app.state.db_engine.dispose()
            logger.info("PostgreSQL connection pool closed")
        except Exception:
            logger.error("PostgreSQL close failed", exc_info=True)

    logger.info("Application shutdown complete", status="stopped")
