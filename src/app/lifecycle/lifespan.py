"""Application lifespan management."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import redis
from celery import Celery
from fastapi import FastAPI
from kombu.exceptions import OperationalError
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from neo4j import AsyncDriver
from neo4j.exceptions import ConfigurationError, ServiceUnavailable
from playwright.async_api import Error as PlaywrightError
from returns.result import Failure

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
from app.features.auth.repository import RefreshTokenRepository
from app.middleware import initialize_fastapi_guard
from app.shared.langchain_layer.agents.memory import setup_cognee
from app.shared.langchain_layer.agents.memory.cognee_client import (
    CogneeDimensionMismatchError,
    CogneeSetupError,
)
from app.shared.langgraph_layer.checkpointer import teardown_langgraph_checkpointer
from app.shared.otel import shutdown_otel
from app.shared.rag.graphiti import close_graphiti, setup_graphiti, setup_graphiti_indices
from app.shared.services.storage import StorageService
from app.utils import ServiceUnavailableException, logger

if TYPE_CHECKING:
    from graphiti_core import Graphiti
    from redis.asyncio.client import Redis


async def setup_redis(url: str) -> redis.asyncio.Redis | None:
    """Initialize Redis with health check."""
    try:
        client: Redis = create_redis_client(url)
        await client.ping()
    except (ConnectionError, TimeoutError, OSError, redis.exceptions.RedisError) as exc:
        logger.warning("Redis startup failed, continuing without cache", error=str(exc))
        return None

    logger.info("Redis connected")
    return client


async def setup_mongodb(
    uri: str, db_name: str, document_models: list[type]
) -> tuple[AsyncIOMotorClient[Any], AsyncIOMotorDatabase[Any]] | None:
    """Initialize MongoDB with health check."""
    try:
        mongo_client, db = await create_mongo_client(
            uri=uri,
            db_name=db_name,
            document_models=document_models,
        )
        await mongo_client.admin.command(command="ping")
        server_info = await mongo_client.server_info()
    except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("MongoDB startup failed, continuing without document store", error=str(exc))
        return None

    logger.info(
        "MongoDB connected",
        database=db_name,
        version=server_info.get("version", "unknown"),
    )
    return mongo_client, db


async def setup_neo4j() -> AsyncDriver | None:
    """Initialize Neo4j with connectivity verification."""
    try:
        neo4j_driver = await init_neo4j()
        await neo4j_driver.verify_connectivity()
    except (
        ConnectionError,
        TimeoutError,
        OSError,
        RuntimeError,
        ValueError,
        ServiceUnavailable,
        ConfigurationError,
    ) as exc:
        logger.warning("Neo4j startup failed, continuing without graph features", error=str(exc))
        return None

    logger.info("Neo4j driver initialized")
    return neo4j_driver


def setup_celery() -> Celery | None:
    """Verify Celery connection to RabbitMQ."""
    try:
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.release()
        logger.info("Celery connected to RabbitMQ")
    except (ServiceUnavailableException, OperationalError, OSError) as e:
        logger.warning("Celery connection failed, tasks will be unavailable", error=str(e))
        return None
    else:
        return celery_app


async def _init_object_storage(app: FastAPI, settings: Any) -> None:
    if settings.S3_BUCKET_NAME:
        app.state.object_store = StorageService.from_settings(settings=settings)
        result = await app.state.object_store.verify_access()
        if isinstance(result, Failure):
            error = result.failure()
            logger.warning(
                "Object storage access verification failed",
                error=error.message,
                details=error.details,
            )
            app.state.object_store = None
            return
        logger.info("Object storage initialized: bucket={}", settings.S3_BUCKET_NAME)
    else:
        app.state.object_store = None
        logger.info("Object storage not configured, skipping")


async def _init_outbox_relay(app: FastAPI, celery_app: Celery | None) -> None:
    from app.connections.postgres import (
        get_database_url,
    )
    from app.shared.outbox import (
        OutboxRelay,
    )

    dsn = get_database_url(flavour="plain")
    relay = OutboxRelay(
        database_url=dsn,
        celery_app=celery_app or app.state.celery,
        session_factory=app.state.db_session_local,
    )
    await relay.run_startup_scan()
    app.state.outbox_relay_task = asyncio.create_task(coro=relay.run_listener())
    app.state.outbox_relay = relay
    logger.info("Outbox relay started")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: PLR0912, PLR0914, PLR0915
    """Manage application startup and shutdown with parallel execution.

    Exception families survived (14 named handlers, single catch-all is `except Exception` for Cognee):
    Redis (ConnectionError, TimeoutError, OSError, redis.exceptions.RedisError),
    MongoDB (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError),
    Neo4j (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, ServiceUnavailable, ConfigurationError),
    Celery (ServiceUnavailableException, OperationalError, OSError),
    TaskGroup ExceptionGroup, PostgreSQL startup (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError),
    CogneeDimensionMismatchError (hard-fail), Graphiti (ConnectionError, TimeoutError, OSError, ServiceUnavailable),
    Crawl4AI (ConnectionError, TimeoutError, OSError, PlaywrightError),
    Object storage (ConnectionError, TimeoutError, OSError), Celery TimeoutError,
    Celery ServiceUnavailableException, Outbox (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError).
    Single `except Exception` for Cognee optional dep remains.
    """
    settings = get_settings()
    logger.info("Application starting", app_name=app.title, version=app.version)

    # STARTUP: Parallel execution for optional services; PostgreSQL remains required for the app to function.
    try:
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
    except ExceptionGroup as exc_group:
        logger.warning(
            "One or more startup tasks failed; continuing with available services",
            error=str(exc_group),
        )
        pg_task = None
        mongo_task = None
        redis_task = None
        neo_task = None

    # Critical dependency: PostgreSQL
    if pg_task is None:
        msg = "PostgreSQL startup failed"
        raise ServiceUnavailableException(msg)

    try:
        app.state.db_engine, app.state.db_session_local = pg_task.result()
    except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
        msg = f"PostgreSQL startup failed: {exc}"
        raise ServiceUnavailableException(msg) from exc

    # Non-critical deps: use None if unavailable
    if mongo_task is not None:
        mongo_result = mongo_task.result()
        if mongo_result is not None:
            app.state.mongo_client, app.state.db = mongo_result
        else:
            app.state.mongo_client = None
            app.state.db = None
    else:
        app.state.mongo_client = None
        app.state.db = None

    if redis_task is not None:
        redis_result = redis_task.result()
        app.state.redis = redis_result
    else:
        app.state.redis = None

    if neo_task is not None:
        neo_result = neo_task.result()
        app.state.neo4j_driver = neo_result
    else:
        app.state.neo4j_driver = None

    # Task 3.1/3.4: the security service needs the token repo so it can
    # re-read session state (pull-based revocation) for active connections.
    ws_redis = getattr(app.state, "redis", None)
    app.state.websocket_security = await build_websocket_security_service(
        redis=ws_redis,
        settings=settings,
        token_repo=RefreshTokenRepository(ws_redis) if ws_redis is not None else None,
    )
    app.state.websocket_revocation_task = asyncio.create_task(
        coro=app.state.websocket_security.run_revocation_loop(),
    )

    # Setup Cognee for episodic + procedural memory (optional)
    try:
        cognee_config = await setup_cognee(settings)
        app.state.cognee_config = cognee_config
        logger.info("Cognee configured")
    except CogneeDimensionMismatchError:
        # Decision 15 hard-fail class: incomparable embedding spaces must stop
        # the boot rather than silently degrade retrieval quality.
        raise
    except CogneeSetupError as exc:
        # Named configuration failure (placeholder/divergent connection
        # settings): degrade without episodic memory, never crash startup.
        exc.add_note("operation=setup_cognee")
        logger.warning("Cognee misconfigured, continuing without episodic memory", error=str(exc))
        app.state.cognee_config = None
    except Exception as exc:  # noqa: BLE001 — optional dependency; app degrades without it
        exc.add_note("operation=setup_cognee")
        logger.warning("Cognee startup failed, continuing without episodic memory", error=str(exc))
        app.state.cognee_config = None

    # Setup Graphiti for legal knowledge graph (optional)
    try:
        graphiti: Graphiti = await setup_graphiti(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USERNAME,
            neo4j_password=settings.NEO4J_PASSWORD.get_secret_value(),
        )
        await setup_graphiti_indices(graphiti)
        app.state.graphiti = graphiti
        logger.info("Graphiti initialized")
    except (ConnectionError, TimeoutError, OSError, ServiceUnavailable) as exc:
        exc.add_note("operation=setup_graphiti")
        logger.warning("Graphiti startup failed, continuing without graph features", error=str(exc))
        app.state.graphiti = None

    # Warn on Neo4j/Graphiti state inconsistency
    neo4j_ok = getattr(app.state, "neo4j_driver", None) is not None
    graphiti_ok = getattr(app.state, "graphiti", None) is not None
    if not neo4j_ok and graphiti_ok:
        logger.warning(
            "State inconsistency: Neo4j driver unavailable but Graphiti initialised independently"
        )
    elif neo4j_ok and not graphiti_ok:
        logger.warning("State inconsistency: Neo4j driver available but Graphiti not initialised")

    # ingestion_llm = ChatGoogleGenerativeAI(
    #     model=settings.GEMINI_FLASH_MODEL,
    #     api_key=settings.GEMINI_API_KEY.get_secret_value(),
    #     temperature=0.1,
    #     retries=0,
    # )
    # app.state.ingestion_graph = build_ingestion_graph(
    #     extraction_llm=ingestion_llm,
    #     db_engine=app.state.db_engine,
    #     graphiti_service=graphiti,
    #     redis=app.state.redis,
    # )
    # NOTE: no `embedding_fn=` here. The graph resolves the embedding client itself, from
    # `app.shared.langchain_layer.embeddings`, and passing one is now a TypeError. If this
    # block is ever uncommented, do not restore the argument from an older revision.
    # logger.info("Contract KB ingestion graph initialized")
    # app.state.pageindex_client = PageIndexClient()
    # Initialize HTTPX client (HTTP/2 + connection pooling)
    app.state.httpx_client = get_shared_httpx_client()
    logger.info("HTTPX client initialized with HTTP/2")
    # Initialize Tavily HTTP client
    app.state.tavily_http_client = await create_tavily_http_client()
    logger.info("Tavily HTTP client initialized")

    # Initialize Crawl4AI browser
    try:
        app.state.crawl4ai_crawler = await create_crawl4ai_crawler()
        logger.info("Crawl4AI browser initialized")
    except (ConnectionError, TimeoutError, OSError, PlaywrightError):
        logger.exception("Crawl4AI browser startup failed, continuing without crawl capability")
        app.state.crawl4ai_crawler = None
    settings = get_settings()
    # Initialize object storage (S3/R2) — optional, graceful degradation
    try:
        await _init_object_storage(app, settings)
    except (ConnectionError, TimeoutError, OSError):
        logger.exception("Object storage startup failed, continuing without")
        app.state.object_store = None

    # Celery setup (optional, non-blocking)
    try:
        celery: Celery | None = await asyncio.wait_for(asyncio.to_thread(setup_celery), timeout=3.0)
        app.state.celery = celery
    except TimeoutError:
        logger.warning("Celery setup timed out, continuing without task queue")
        app.state.celery = None
    except ServiceUnavailableException as e:
        logger.error("Celery setup failed", error=str(e))
        app.state.celery = None

    # Outbox relay (uses existing database session factory)
    try:
        await _init_outbox_relay(app, celery_app)
    except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
        exc.add_note("operation=setup_outbox_relay")
        logger.warning("Outbox relay startup failed, continuing without outbox", error=str(exc))
        app.state.outbox_relay = None

    # FastAPI-Guard setup (depends on Redis, but non-blocking)
    await initialize_fastapi_guard(app=app, settings=settings)

    # LangGraph checkpointer setup (uses existing PostgreSQL connection).
    # Deliberately left unwired. If it is ever re-enabled: the checkpointer is
    # psycopg-backed, so it needs the plain flavour of the accessor -- a raw
    # settings.POSTGRES_URL carries no credential, and the async flavour carries a
    # dialect scheme psycopg cannot parse.
    #     from app.connections.postgres import get_database_url
    # try:
    #     saul_checkpointer = await setup_langgraph_checkpointer(
    #         conn_string=get_database_url(flavour="plain"),
    #     )
    #     app.state.langgraph_checkpointer = saul_checkpointer
    #     logger.info("LangGraph checkpointer initialized")
    # except (ConnectionError, TimeoutError, OSError) as e:
    #     logger.error(
    #         "LangGraph checkpointer setup failed, continuing without persistence", error=str(e)
    #     )
    #     app.state.langgraph_checkpointer = None

    logger.info("Application ready", status="running")

    try:
        yield
    finally:
        # SHUTDOWN: Parallel graceful cleanup
        logger.info("Application shutting down", status="stopping")

        # Close LangGraph checkpointer connection pool
        if hasattr(app.state, "langgraph_checkpointer"):
            await teardown_langgraph_checkpointer(app.state.langgraph_checkpointer)

        # Stop outbox relay
        if hasattr(app.state, "outbox_relay_task") and app.state.outbox_relay_task is not None:
            app.state.outbox_relay_task.cancel()
            logger.info("Outbox relay stopped")

        # Stop WebSocket revocation loop
        revocation_task = getattr(app.state, "websocket_revocation_task", None)
        if revocation_task is not None:
            revocation_task.cancel()
            logger.info("WebSocket revocation loop stopped")

        # Close HTTPX client
        httpx_client = getattr(app.state, "httpx_client", None)
        if httpx_client is not None:
            await httpx_client.aclose()

        # Close Tavily HTTP client
        tavily_http_client = getattr(app.state, "tavily_http_client", None)
        if tavily_http_client is not None:
            await close_tavily_http_client(tavily_http_client)

        if hasattr(app.state, "graphiti"):
            await close_graphiti(app.state.graphiti)

        if hasattr(app.state, "crawl4ai_crawler"):
            await close_crawl4ai_crawler(app.state.crawl4ai_crawler)

        # MongoDB close is synchronous - run outside TaskGroup
        mongo_client = getattr(app.state, "mongo_client", None)
        if mongo_client is not None:
            mongo_client.close()

        async with asyncio.TaskGroup() as tg:
            redis_client = getattr(app.state, "redis", None)
            if redis_client is not None:
                tg.create_task(coro=redis_client.aclose(close_connection_pool=True))

            db_engine = getattr(app.state, "db_engine", None)
            if db_engine is not None:
                tg.create_task(coro=db_engine.dispose())

            neo4j_driver = getattr(app.state, "neo4j_driver", None)
            if neo4j_driver is not None:
                tg.create_task(coro=close_neo4j_driver(driver=neo4j_driver))
        # Shutdown OpenTelemetry (flush remaining spans/metrics/logs)
        shutdown_otel()

        logger.info("Application shutdown complete", status="stopped")
