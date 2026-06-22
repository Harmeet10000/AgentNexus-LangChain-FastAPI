"""Application lifespan management."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import redis
from celery import Celery
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from neo4j import AsyncDriver

from app.config import get_settings
from app.connections import (
    celery_app,
    close_crawl4ai_crawler,
    close_neo4j_driver,
    close_tavily_http_client,
    create_crawl4ai_crawler,
    create_mongo_client,
    create_redis_client,
    create_tavily_http_client,
    get_shared_httpx_client,
    init_db,
    init_neo4j,
)
from app.features.auth import TokenAuditLog, User, build_websocket_security_service
from app.features.documents import run_document_startup_checks
from app.middleware import initialize_fastapi_guard
from app.shared.langchain_layer.agents.memory import setup_cognee
from app.shared.langgraph_layer.checkpointer import teardown_langgraph_checkpointer
from app.shared.rag.graphiti import close_graphiti, setup_graphiti, setup_graphiti_indices
from app.shared.services.storage import StorageService
from app.utils import ServiceUnavailableException, logger
from mcp_core import get_mcp_client_manager

if TYPE_CHECKING:
    from graphiti_core import Graphiti
    from httpx._client import AsyncClient

    from app.features.auth.websocket_security import WebSocketSecurityService


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
    except ServiceUnavailableException as e:
        logger.warning("Celery connection failed, tasks will be unavailable", error=str(e))
        return None
    else:
        return celery_app


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: PLR0915, PLR0912
    """Manage application startup and shutdown with parallel execution."""
    settings = get_settings()
    logger.info("Application starting", app_name=app.title, version=app.version)

    # STARTUP: Parallel execution — critical deps fail-fast, optional deps degrade
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

    # Critical deps: PG, Redis, MongoDB — raise on failure
    try:
        app.state.db_engine, app.state.db_session_local = pg_task.result()
    except Exception as exc:
        msg = f"PostgreSQL startup failed: {exc}"
        raise ServiceUnavailableException(msg) from exc

    try:
        app.state.mongo_client, app.state.db = mongo_task.result()
    except Exception as exc:
        msg = f"MongoDB startup failed: {exc}"
        raise ServiceUnavailableException(msg) from exc

    try:
        app.state.redis: redis.asyncio.Redis = redis_task.result()
    except Exception as exc:
        msg = f"Redis startup failed: {exc}"
        raise ServiceUnavailableException(msg) from exc

    # Optional deps: Neo4j, Graphiti — log warning, set None on failure
    try:
        app.state.neo4j_driver: AsyncDriver = neo_task.result()
    except Exception as exc:  # noqa: BLE001 — graceful degradation for optional dep
        logger.warning("Neo4j startup failed, continuing without graph features", error=str(exc))
        app.state.neo4j_driver = None  # type: ignore[assignment]

    app.state.websocket_security: WebSocketSecurityService = await build_websocket_security_service(
        redis=app.state.redis,
        settings=settings,
    )

    # Setup Cognee for episodic + procedural memory
    cognee_config = await setup_cognee(settings)
    app.state.cognee_config = cognee_config
    logger.info("Cognee configured")

    # Setup Graphiti for legal knowledge graph (optional)
    try:
        graphiti = await setup_graphiti(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USERNAME,
            neo4j_password=settings.NEO4J_PASSWORD.get_secret_value(),
        )
        await setup_graphiti_indices(graphiti)
        app.state.graphiti: Graphiti = graphiti
        logger.info("Graphiti initialized")
    except Exception as exc:  # noqa: BLE001 — graceful degradation for optional dep
        logger.warning("Graphiti startup failed, continuing without graph features", error=str(exc))
        app.state.graphiti = None  # type: ignore[assignment]

    # ingestion_llm = ChatGoogleGenerativeAI(
    #     model=settings.GEMINI_FLASH_MODEL,
    #     api_key=settings.GEMINI_API_KEY.get_secret_value(),
    #     temperature=0.1,
    #     retries=0,
    # )
    # app.state.ingestion_graph = build_ingestion_graph(
    #     extraction_llm=ingestion_llm,
    #     db_engine=app.state.db_engine,
    #     embedding_fn=build_embedding_client(),
    #     graphiti_service=graphiti,
    #     redis=app.state.redis,
    # )
    # logger.info("Contract KB ingestion graph initialized")
    # app.state.pageindex_client = PageIndexClient()
    # Initialize HTTPX client (HTTP/2 + connection pooling)
    app.state.httpx_client: AsyncClient = get_shared_httpx_client()
    logger.info("HTTPX client initialized with HTTP/2")
    # Initialize Tavily HTTP client
    app.state.tavily_http_client: AsyncClient = await create_tavily_http_client()
    logger.info("Tavily HTTP client initialized")

    # Initialize Crawl4AI browser
    try:
        app.state.crawl4ai_crawler = await create_crawl4ai_crawler()
        logger.info("Crawl4AI browser initialized")
    except Exception:  # noqa: BLE001 — graceful degradation for optional dep
        logger.exception("Crawl4AI browser startup failed, continuing without crawl capability")
        app.state.crawl4ai_crawler = None
    settings = get_settings()
    app.state.object_store = StorageService.from_settings(settings=settings)
    await run_document_startup_checks(
        engine=app.state.db_engine,
        object_store=app.state.object_store,
    )

    # Celery setup (optional, non-blocking)
    try:
        celery: Celery | None = await asyncio.wait_for(asyncio.to_thread(setup_celery), timeout=3.0)
        app.state.celery: Celery | None = celery
    except TimeoutError:
        logger.warning("Celery setup timed out, continuing without task queue")
        app.state.celery = None
    except ServiceUnavailableException as e:
        logger.error("Celery setup failed", error=str(e))
        app.state.celery = None

    # Outbox relay (uses existing database session factory)
    try:
        from app.connections.postgres import get_database_url  # noqa: PLC0415
        from app.shared.outbox import OutboxRelay  # noqa: PLC0415

        dsn = get_database_url().replace("+asyncpg", "")
        relay = OutboxRelay(
            database_url=dsn,
            celery_app=celery_app or app.state.celery,
            session_factory=app.state.db_session_local,
        )
        await relay.run_startup_scan()
        app.state.outbox_relay_task = asyncio.create_task(relay.run_listener())
        app.state.outbox_relay = relay
        logger.info("Outbox relay started")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Outbox relay startup failed, continuing without outbox", error=str(exc))
        app.state.outbox_relay = None

    # FastAPI-Guard setup (depends on Redis, but non-blocking)
    await initialize_fastapi_guard(app=app, settings=settings)

    # LangGraph checkpointer setup (uses existing PostgreSQL connection)
    # try:
    #     saul_checkpointer = await setup_langgraph_checkpointer(
    #         conn_string=settings.POSTGRES_URL,
    #     )
    #     app.state.langgraph_checkpointer = saul_checkpointer
    #     logger.info("LangGraph checkpointer initialized")
    # except (ConnectionError, TimeoutError, OSError) as e:
    #     logger.error(
    #         "LangGraph checkpointer setup failed, continuing without persistence", error=str(e)
    #     )
    #     app.state.langgraph_checkpointer = None

    # MCPClientManager lifecycle (lazy-connects to upstream MCP servers)
    try:
        app.state.mcp_manager = get_mcp_client_manager()
        logger.info("MCP client manager initialized")
    except Exception:  # noqa: BLE001
        logger.exception("MCP client manager startup failed, continuing without MCP tools")
        app.state.mcp_manager = None

    logger.info("Application ready", status="running")

    yield

    # SHUTDOWN: Parallel graceful cleanup
    logger.info("Application shutting down", status="stopping")

    # Close LangGraph checkpointer connection pool
    if hasattr(app.state, "langgraph_checkpointer"):
        await teardown_langgraph_checkpointer(app.state.langgraph_checkpointer)

    # Stop outbox relay
    if hasattr(app.state, "outbox_relay_task") and app.state.outbox_relay_task is not None:
        app.state.outbox_relay_task.cancel()
        logger.info("Outbox relay stopped")

    # Close HTTPX client
    if hasattr(app.state, "httpx_client"):
        await app.state.httpx_client.aclose()

    # Close Tavily HTTP client
    if hasattr(app.state, "tavily_http_client"):
        await close_tavily_http_client(app.state.tavily_http_client)

    if hasattr(app.state, "graphiti"):
        await close_graphiti(app.state.graphiti)

    if hasattr(app.state, "crawl4ai_crawler"):
        await close_crawl4ai_crawler(app.state.crawl4ai_crawler)

    # MongoDB close is synchronous - run outside TaskGroup
    if hasattr(app.state, "mongo_client"):
        app.state.mongo_client.close()

    async with asyncio.TaskGroup() as tg:
        if hasattr(app.state, "redis"):
            tg.create_task(coro=app.state.redis.aclose(close_connection_pool=True))
        if hasattr(app.state, "db_engine"):
            tg.create_task(coro=app.state.db_engine.dispose())
        if hasattr(app.state, "neo4j_driver"):
            tg.create_task(coro=close_neo4j_driver(driver=app.state.neo4j_driver))
        if hasattr(app.state, "mcp_manager") and app.state.mcp_manager is not None:
            tg.create_task(coro=app.state.mcp_manager.close())

    logger.info("Application shutdown complete", status="stopped")
