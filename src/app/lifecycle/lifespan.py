"""Application lifespan management."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

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
from app.features.health.health_check import ALL_PROBES, check_cognee, check_graphiti
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
from app.utils import DependencyHealth, ServiceUnavailableException, logger

if TYPE_CHECKING:
    from graphiti_core import Graphiti
    from redis.asyncio.client import Redis


async def setup_redis(url: str) -> redis.asyncio.Redis | None:
    """Initialize Redis with health check."""
    try:
        client: Redis = create_redis_client(url)
        await client.ping()
    except (ConnectionError, TimeoutError, OSError, redis.exceptions.RedisError) as exc:
        logger.bind(component="redis", error=str(exc)).warning(
            "Redis startup failed, continuing without cache"
        )
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
        logger.bind(component="mongodb", error=str(exc)).warning(
            "MongoDB startup failed, continuing without document store"
        )
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
        logger.bind(component="neo4j", error=str(exc)).warning(
            "Neo4j startup failed, continuing without graph features"
        )
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
        logger.bind(component="celery", error=str(e)).warning(
            "Celery connection failed, tasks will be unavailable"
        )
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
        logger.bind(bucket=settings.S3_BUCKET_NAME).info("Object storage initialized")
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


class StartupPolicy(NamedTuple):
    """One optional-dependency boot block as data.

    `setup` does the work against `app.state`; `fatal_on` types propagate out
    of boot; `degrade_on` types are reported via `report` and land
    `state_attr` on None. `probe` links the policies whose dependency maps 1:1
    to a deep-health probe to that same function object in
    `health_check.ALL_PROBES` (the canonical dependency inventory) instead of
    restating the name; policies with no 1:1 probe leave it None.
    """

    name: str
    setup: Callable[[FastAPI, Any], Awaitable[None]]
    state_attr: str
    fatal_on: tuple[type[BaseException], ...]
    degrade_on: tuple[type[BaseException], ...]
    report: Callable[[BaseException], None]
    probe: Callable[[FastAPI], Awaitable[DependencyHealth]] | None = None


async def _setup_cognee_state(app: FastAPI, settings: Any) -> None:
    """Configure episodic memory; the hard-fail class propagates to the runner."""
    app.state.cognee_config = await setup_cognee(settings)
    logger.info("Cognee configured")


async def _setup_graphiti_state(app: FastAPI, settings: Any) -> None:
    """Initialise the legal knowledge graph and its indices."""
    graphiti: Graphiti = await setup_graphiti(
        neo4j_uri=settings.NEO4J_URI,
        neo4j_user=settings.NEO4J_USERNAME,
        neo4j_password=settings.NEO4J_PASSWORD.get_secret_value(),
    )
    await setup_graphiti_indices(graphiti)
    app.state.graphiti = graphiti
    logger.info("Graphiti initialized")


async def _setup_crawl4ai_state(app: FastAPI, _settings: Any) -> None:
    """Initialise the Crawl4AI browser."""
    from app.shared.crawler.processor import get_processor

    app.state.crawl4ai_crawler = await create_crawl4ai_crawler()
    app.state.crawler_processor = await get_processor()
    logger.info("Crawl4AI browser initialized")


async def _setup_object_storage_state(app: FastAPI, settings: Any) -> None:
    """Initialise object storage; access-verification failure degrades inside."""
    await _init_object_storage(app, settings)


async def _setup_celery_state(app: FastAPI, _settings: Any) -> None:
    """Verify the Celery/RabbitMQ connection without blocking boot."""
    celery: Celery | None = await asyncio.wait_for(asyncio.to_thread(setup_celery), timeout=3.0)
    app.state.celery = celery


async def _setup_outbox_relay_state(app: FastAPI, _settings: Any) -> None:
    """Start the outbox relay listener on the existing session factory."""
    await _init_outbox_relay(app, celery_app)


def _report_cognee_degraded(exc: BaseException) -> None:
    """Cognee degrade path: misconfiguration and unexpected failure log differently."""
    exc.add_note("operation=setup_cognee")
    if isinstance(exc, CogneeSetupError):
        logger.bind(component="cognee", operation="setup_cognee", error=str(exc)).warning(
            "Cognee misconfigured, continuing without episodic memory"
        )
    else:
        logger.bind(component="cognee", operation="setup_cognee", error=str(exc)).warning(
            "Cognee startup failed, continuing without episodic memory"
        )


def _report_graphiti_degraded(exc: BaseException) -> None:
    """Graphiti degrade path."""
    exc.add_note("operation=setup_graphiti")
    logger.bind(component="graphiti", operation="setup_graphiti", error=str(exc)).warning(
        "Graphiti startup failed, continuing without graph features"
    )


def _report_crawl4ai_degraded(_exc: BaseException) -> None:
    """Crawl4AI degrade path: keeps the original `logger.exception` shape."""
    logger.exception("Crawl4AI browser startup failed, continuing without crawl capability")


def _report_object_storage_degraded(_exc: BaseException) -> None:
    """Object-storage degrade path: keeps the original `logger.exception` shape."""
    logger.exception("Object storage startup failed, continuing without")


def _report_celery_degraded(exc: BaseException) -> None:
    """Celery degrade path: a slow broker and a refusing broker log differently."""
    if isinstance(exc, TimeoutError):
        logger.warning("Celery setup timed out, continuing without task queue")
    else:
        logger.bind(component="celery", error=str(exc)).exception("Celery setup failed")


def _report_outbox_relay_degraded(exc: BaseException) -> None:
    """Outbox-relay degrade path."""
    exc.add_note("operation=setup_outbox_relay")
    logger.bind(component="outbox_relay", operation="setup_outbox_relay", error=str(exc)).warning(
        "Outbox relay startup failed, continuing without outbox"
    )


# Optional-dependency boot order. Each entry replaces one hand-rolled
# try/except-degrade block: adding an optional dependency is one entry here
# plus its two small functions — no new try/except inside `lifespan`.
STARTUP_POLICIES: tuple[StartupPolicy, ...] = (
    StartupPolicy(
        name="cognee",
        setup=_setup_cognee_state,
        state_attr="cognee_config",
        fatal_on=(CogneeDimensionMismatchError,),
        # Single catch-all for the optional dep: anything that is not the
        # hard-fail class degrades without episodic memory.
        degrade_on=(Exception,),
        report=_report_cognee_degraded,
        probe=check_cognee,
    ),
    StartupPolicy(
        name="graphiti",
        setup=_setup_graphiti_state,
        state_attr="graphiti",
        fatal_on=(),
        degrade_on=(ConnectionError, TimeoutError, OSError, ServiceUnavailable),
        report=_report_graphiti_degraded,
        probe=check_graphiti,
    ),
    StartupPolicy(
        name="crawl4ai",
        setup=_setup_crawl4ai_state,
        state_attr="crawl4ai_crawler",
        fatal_on=(),
        degrade_on=(ConnectionError, TimeoutError, OSError, PlaywrightError),
        report=_report_crawl4ai_degraded,
    ),
    StartupPolicy(
        name="object_storage",
        setup=_setup_object_storage_state,
        state_attr="object_store",
        fatal_on=(),
        degrade_on=(ConnectionError, TimeoutError, OSError),
        report=_report_object_storage_degraded,
    ),
    StartupPolicy(
        name="celery",
        setup=_setup_celery_state,
        state_attr="celery",
        fatal_on=(),
        degrade_on=(TimeoutError, ServiceUnavailableException),
        report=_report_celery_degraded,
    ),
    StartupPolicy(
        name="outbox_relay",
        setup=_setup_outbox_relay_state,
        state_attr="outbox_relay",
        fatal_on=(),
        degrade_on=(ConnectionError, TimeoutError, OSError, RuntimeError, ValueError),
        report=_report_outbox_relay_degraded,
    ),
)

_POLICY_PROBES = frozenset(policy.probe for policy in STARTUP_POLICIES if policy.probe is not None)
if not _POLICY_PROBES.issubset(ALL_PROBES):
    msg = "STARTUP_POLICIES probes must be members of health_check.ALL_PROBES"
    raise RuntimeError(msg)


async def _run_startup_policy(app: FastAPI, settings: Any, policy: StartupPolicy) -> None:
    """Run one boot policy: fatal types propagate, degrade types degrade in place.

    The `fatal_on` clause comes first deliberately: `CogneeDimensionMismatchError`
    subclasses `CogneeSetupError`, which the cognee policy also degrades on — a
    broad-first order would swallow the one hard-fail class and silently degrade
    retrieval quality instead of stopping the boot.
    """
    try:
        await policy.setup(app, settings)
    except policy.fatal_on:
        raise
    except policy.degrade_on as exc:
        policy.report(exc)
        setattr(app.state, policy.state_attr, None)


async def _shutdown_resources(app: FastAPI) -> None:  # noqa: PLR0912
    """Close application resources while always flushing observability providers."""
    try:
        if hasattr(app.state, "langgraph_checkpointer"):
            await teardown_langgraph_checkpointer(app.state.langgraph_checkpointer)

        if hasattr(app.state, "outbox_relay_task") and app.state.outbox_relay_task is not None:
            app.state.outbox_relay_task.cancel()
            logger.info("Outbox relay stopped")

        revocation_task = getattr(app.state, "websocket_revocation_task", None)
        if revocation_task is not None:
            revocation_task.cancel()
            logger.info("WebSocket revocation loop stopped")

        websocket_security = getattr(app.state, "websocket_security", None)
        if websocket_security is not None:
            websocket_security.close()
            logger.info("WebSocket rate limiters closed")

        httpx_client = getattr(app.state, "httpx_client", None)
        if httpx_client is not None:
            await httpx_client.aclose()

        tavily_http_client = getattr(app.state, "tavily_http_client", None)
        if tavily_http_client is not None:
            await close_tavily_http_client(tavily_http_client)

        if hasattr(app.state, "graphiti"):
            await close_graphiti(app.state.graphiti)

        if hasattr(app.state, "crawl4ai_crawler"):
            await close_crawl4ai_crawler(app.state.crawl4ai_crawler)

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
    finally:
        logger.bind(status="stopped").info("Application shutdown complete")
        shutdown_otel()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: PLR0912, PLR0915
    """Manage application startup and shutdown with parallel execution.

    Exception families survived (per-policy degrade tuples in `STARTUP_POLICIES`;
    single catch-all is the cognee policy's `degrade_on=(Exception,)`):
    Redis (ConnectionError, TimeoutError, OSError, redis.exceptions.RedisError),
    MongoDB (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError),
    Neo4j (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError, ServiceUnavailable, ConfigurationError),
    Celery (ServiceUnavailableException, OperationalError, OSError),
    TaskGroup ExceptionGroup, PostgreSQL startup (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError),
    CogneeDimensionMismatchError (hard-fail), Graphiti (ConnectionError, TimeoutError, OSError, ServiceUnavailable),
    Crawl4AI (ConnectionError, TimeoutError, OSError, PlaywrightError),
    Object storage (ConnectionError, TimeoutError, OSError), Celery TimeoutError,
    Celery ServiceUnavailableException, Outbox (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError).
    """
    settings = get_settings()
    logger.bind(app_name=app.title, version=app.version).info("Application starting")

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
        logger.bind(error=str(exc_group)).warning(
            "One or more startup tasks failed, continuing with available services"
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

    # Optional dependencies boot through the policy registry in order. The
    # Neo4j/Graphiti consistency warnings below stay inline: they cross-cut two
    # policies (the TaskGroup neo4j result and the graphiti policy outcome) and
    # are a check, not a setup — registry form would force behaviour change.
    for policy in STARTUP_POLICIES:
        await _run_startup_policy(app, settings, policy)

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

    logger.bind(status="running").info("Application ready")

    try:
        yield
    finally:
        logger.bind(status="stopping").info("Application shutting down")
        await _shutdown_resources(app)
