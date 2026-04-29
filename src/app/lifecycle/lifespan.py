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
    close_neo4j_driver,
    close_tavily_http_client,
    create_mongo_client,
    create_redis_client,
    create_tavily_http_client,
    get_shared_httpx_client,
    init_db,
    init_neo4j,
)
from app.features.auth import TokenAuditLog, User, build_websocket_security_service
from app.features.search.embeddings import build_embedding_client
from app.middleware import initialize_fastapi_guard

# from app.shared import get_mcp_client_manager
from app.shared.langchain_layer.agents.memory import setup_cognee
from app.shared.langgraph_layer.checkpointer import (
    setup_langgraph_checkpointer,
    teardown_langgraph_checkpointer,
)
from app.shared.langgraph_layer.ingestion_kb import build_ingestion_graph
from app.shared.rag.graphiti import close_graphiti, setup_graphiti, setup_graphiti_indices
from app.utils import ServiceUnavailableException, logger

if TYPE_CHECKING:
    from graphiti_core import Graphiti
    from app.features.auth.websocket_security import WebSocketSecurityService
    # from app.shared.mcp import MCPClientManager


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
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: PLR0915
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
    app.state.redis: redis.asyncio.Redis = redis_task.result()
    app.state.neo4j_driver: AsyncDriver = neo_task.result()
    app.state.websocket_security: WebSocketSecurityService = await build_websocket_security_service(
        redis=app.state.redis,
        settings=settings,
    )

    # Setup Cognee for episodic + procedural memory
    cognee_config = await setup_cognee(settings)
    app.state.cognee_config = cognee_config
    logger.info("Cognee configured")

    # Setup Graphiti for legal knowledge graph
    graphiti = await setup_graphiti(
        neo4j_uri=settings.NEO4J_URI,
        neo4j_user=settings.NEO4J_USERNAME,
        neo4j_password=settings.NEO4J_PASSWORD,
    )
    await setup_graphiti_indices(graphiti)
    app.state.graphiti: Graphiti = graphiti
    logger.info("Graphiti initialized")

    # ingestion_llm = ChatGoogleGenerativeAI(
    #     model=settings.GEMINI_FLASH_MODEL,
    #     api_key=settings.GEMINI_API_KEY,
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
    app.state.httpx_client = get_shared_httpx_client()
    logger.info("HTTPX client initialized with HTTP/2")
    # Initialize Tavily HTTP client
    app.state.tavily_http_client = await create_tavily_http_client()
    logger.info("Tavily HTTP client initialized")
    # app.state.storage = StorageService.from_settings()

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

    logger.info("Application ready", status="running")

    yield

    # SHUTDOWN: Parallel graceful cleanup
    logger.info("Application shutting down", status="stopping")

    # Close LangGraph checkpointer connection pool
    if hasattr(app.state, "langgraph_checkpointer"):
        await teardown_langgraph_checkpointer(app.state.langgraph_checkpointer)

    # Close HTTPX client
    if hasattr(app.state, "httpx_client"):
        await app.state.httpx_client.aclose()

    # Close Tavily HTTP client
    if hasattr(app.state, "tavily_http_client"):
        await close_tavily_http_client(app.state.tavily_http_client)

    if hasattr(app.state, "graphiti"):
        await close_graphiti(app.state.graphiti)

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
        # mcp_manager: MCPClientManager = get_mcp_client_manager()
        # if mcp_manager is not None:
            # tg.create_task(coro=mcp_manager.close())

    logger.info("Application shutdown complete", status="stopped")
