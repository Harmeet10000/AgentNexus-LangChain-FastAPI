"""Application lifespan management."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis
from celery import Celery
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from neo4j import AsyncDriver

from app.config import get_settings
from app.connections import (
    celery_app,
    close_neo4j_driver,
    create_mongo_client,
    create_redis_client,
    get_shared_httpx_client,
    init_db,
    init_neo4j,
)
from app.features.auth import TokenAuditLog, User
from app.features.auth.websocket_security import build_websocket_security_service
from app.middleware import initialize_fastapi_guard
from app.shared.mcp import get_mcp_client_manager
from app.utils import logger


async def setup_redis(url: str) -> redis.asyncio.Redis:
    """Initialize Redis with health check."""
    redis = create_redis_client(url)
    await redis.ping()
    logger.info("Redis connected")
    return redis


async def setup_mongodb(
    uri: str, db_name: str, document_models: list
) -> tuple[AsyncIOMotorClient, AsyncIOMotorDatabase]:
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


async def setup_neo4j() -> AsyncDriver:
    """Initialize Neo4j with connectivity verification."""
    neo4j_driver = await init_neo4j()
    await neo4j_driver.verify_connectivity()
    logger.info("Neo4j driver initialized")
    return neo4j_driver


def setup_celery() -> Celery | None:
    """Verify Celery connection to RabbitMQ."""
    try:
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.release()
        logger.info("Celery connected to RabbitMQ")
    except Exception as e:
        logger.warning("Celery connection failed, tasks will be unavailable", error=str(e))
        return None
    else:
        return celery_app


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
                document_models=[User, TokenAuditLog],
            )
        )
        redis_task = tg.create_task(coro=setup_redis(url=settings.REDIS_URL))
        neo_task = tg.create_task(coro=setup_neo4j())

    # All tasks succeeded - store results
    app.state.db_engine, app.state.db_session_local = pg_task.result()
    app.state.mongo_client, app.state.db = mongo_task.result()
    app.state.redis = redis_task.result()
    app.state.neo4j_driver = neo_task.result()
    app.state.websocket_security = build_websocket_security_service(
        redis=app.state.redis,
        settings=settings,
    )
    # app.state.pageindex_client = PageIndexClient()
    # Initialize HTTPX client (HTTP/2 + connection pooling)
    app.state.httpx_client = get_shared_httpx_client()
    logger.info("HTTPX client initialized with HTTP/2")
    # app.state.storage = StorageService.from_settings()

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

    await initialize_fastapi_guard(app=app, settings=settings)

    logger.info("Application ready", status="running")

    yield

    # SHUTDOWN: Parallel graceful cleanup
    logger.info("Application shutting down", status="stopping")

    # Close HTTPX client
    if hasattr(app.state, "httpx_client"):
        await app.state.httpx_client.aclose()

    async with asyncio.TaskGroup() as tg:
        if hasattr(app.state, "redis"):
            tg.create_task(coro=app.state.redis.aclose())
        if hasattr(app.state, "db_engine"):
            tg.create_task(coro=app.state.db_engine.dispose())
        if hasattr(app.state, "neo4j_driver"):
            tg.create_task(coro=close_neo4j_driver(driver=app.state.neo4j_driver))
        mcp_manager = get_mcp_client_manager()
        if mcp_manager is not None:
            tg.create_task(coro=mcp_manager.close())

    # MongoDB close is synchronous - run outside TaskGroup
    if hasattr(app.state, "mongo_client"):
        app.state.mongo_client.close()

    logger.info("Application shutdown complete", status="stopped")
