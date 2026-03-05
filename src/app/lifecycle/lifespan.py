"""Application lifespan management."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.connections import (
    celery_app,
    close_neo4j_driver,
    create_httpx_client,
    create_mongo_client,
    create_redis_client,
    init_db,
    init_neo4j,
)
from app.features.auth import User
from app.utils import logger


async def setup_redis(url: str):
    """Initialize Redis with health check."""
    redis = create_redis_client(url)
    await redis.ping()
    logger.info("Redis connected")
    return redis


async def setup_mongodb(uri: str, db_name: str, document_models: list):
    """Initialize MongoDB with health check."""
    mongo_client, db = await create_mongo_client(
        uri=uri,
        db_name=db_name,
        document_models=document_models,
    )
    await mongo_client.admin.command(command="ping")
    server_info = await mongo_client.server_info()
    logger.info(
        "MongoDB connected",
        database=db_name,
        version=server_info.get("version", "unknown"),
    )
    return mongo_client, db


async def setup_neo4j():
    """Initialize Neo4j with connectivity verification."""
    neo4j_driver = await init_neo4j()
    neo4j_driver.verify_connectivity()
    logger.info("Neo4j driver initialized")
    return neo4j_driver


def setup_celery():
    """Verify Celery connection to RabbitMQ."""
    try:
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.release()
        logger.info("Celery connected to RabbitMQ")
        return celery_app
    except Exception as e:
        logger.warning("Celery connection failed, tasks will be unavailable", error=str(e))
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown with parallel execution."""
    settings = get_settings()
    logger.info("Application starting", app_name=app.title, version=app.version)

    # STARTUP: Parallel execution using TaskGroup (fail-fast)
    async with asyncio.TaskGroup() as tg:
        pg_task = tg.create_task(coro=init_db())
        mongo_task = tg.create_task(
            coro=setup_mongodb(
                uri=settings.MONGODB_URI,
                db_name=settings.MONGODB_DB_NAME,
                document_models=[User],
            )
        )
        redis_task = tg.create_task(coro=setup_redis(url=settings.REDIS_URL))
        neo_task = tg.create_task(coro=setup_neo4j())

    # All tasks succeeded - store results
    app.state.db_engine, app.state.db_session_local = pg_task.result()
    app.state.mongo_client, app.state.db = mongo_task.result()
    app.state.redis = redis_task.result()
    app.state.neo4j_driver = neo_task.result()

    # Initialize HTTPX client (HTTP/2 + connection pooling)
    app.state.httpx_client = create_httpx_client()
    logger.info("HTTPX client initialized with HTTP/2")



    # Celery setup (optional, non-blocking)
    try:
        celery = await asyncio.wait_for(asyncio.to_thread(setup_celery), timeout=3.0)
        app.state.celery = celery
    except TimeoutError:
        logger.warning("Celery setup timed out, continuing without task queue")
        app.state.celery = None
    except Exception as e:
        logger.error("Celery setup failed", error=str(e))
        app.state.celery = None

    logger.info("Application ready", status="running")

    yield

    # SHUTDOWN: Parallel graceful cleanup
    logger.info("Application shutting down", status="stopping")

    # Close HTTPX client
    if hasattr(app.state, "httpx_client"):
        await app.state.httpx_client.aclose()

    async with asyncio.TaskGroup() as tg:
        if hasattr(app.state, "redis"):
            tg.create_task(coro=app.state.redis.close())
        if hasattr(app.state, "db_engine"):
            tg.create_task(coro=app.state.db_engine.dispose())
        if hasattr(app.state, "neo4j_driver"):
            tg.create_task(coro=close_neo4j_driver(driver=app.state.neo4j_driver))

    # MongoDB close is synchronous - run outside TaskGroup
    if hasattr(app.state, "mongo_client"):
        app.state.mongo_client.close()

    logger.info("Application shutdown complete", status="stopped")
