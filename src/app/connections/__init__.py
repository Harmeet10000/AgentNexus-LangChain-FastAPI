"""Database connection dependencies."""

from app.connections.checkpointer import teardown_langgraph_checkpointer
from app.connections.graphiti import (
    close_graphiti,
    setup_graphiti,
    setup_graphiti_indices,
)

from .celery import (
    CircuitBreakerOpenError,
    ResilientTask,
    acquire_idempotency_lock,
    build_circuit_breaker_key,
    build_idempotency_key,
    celery_app,
    create_celery_app,
    get_circuit_breaker_state,
    get_idempotency_status,
    is_circuit_breaker_open,
    mark_idempotency_completed,
    mark_idempotency_failed_permanently,
    record_circuit_breaker_failure,
    record_circuit_breaker_success,
    release_idempotency_processing_lock,
    run_redis_call,
    run_with_circuit_breaker,
    set_circuit_breaker_state,
)
from .cognee import (
    CogneeDimensionMismatchError,
    CogneeSetupConfig,
    CogneeSetupError,
    setup_cognee,
)
from .crawl4ai import (
    close_crawl4ai_crawler,
    create_crawl4ai_crawler,
    get_crawl4ai_crawler,
    get_crawler,
)
from .httpx_client import (
    close_httpx_client,
    create_httpx_client,
    get_httpx_client,
    get_shared_httpx_client,
)
from .mongodb import close_mongo_client, create_mongo_client, get_mongodb
from .neo4j import (
    close_neo4j_driver,
    get_neo4j_driver,
    get_neo4j_session,
    init_neo4j,
)
from .object_storage import create_object_store
from .outbox import create_outbox_relay
from .postgres import close_db_engine, get_postgres_db, init_db
from .redis import close_redis_client, create_redis_client, get_redis
from .tavily import (
    close_tavily_http_client,
    create_tavily_http_client,
    get_shared_tavily_http_client,
    get_tavily_http_client,
)

__all__ = [
    "CogneeDimensionMismatchError",
    "CogneeSetupConfig",
    "CogneeSetupError",
    "ResilientTask",
    "celery_app",
    "close_crawl4ai_crawler",
    "close_db_engine",
    "    close_graphiti",
    "close_httpx_client",
    "close_mongo_client",
    "close_neo4j_driver",
    "close_redis_client",
    "close_tavily_http_client",
    "create_celery_app",
    "create_crawl4ai_crawler",
    "create_httpx_client",
    "create_mongo_client",
    "create_object_store",
    "create_outbox_relay",
    "create_redis_client",
    "create_tavily_http_client",
    "get_crawl4ai_crawler",
    "get_crawler",
    "get_httpx_client",
    "get_mongodb",
    "get_neo4j_driver",
    "get_neo4j_session",
    "get_postgres_db",
    "get_redis",
    "get_shared_httpx_client",
    "get_shared_tavily_http_client",
    "get_tavily_http_client",
    "init_db",
    "init_neo4j",
    "setup_cognee",
    "setup_graphiti",
    "setup_graphiti_indices",
    "teardown_langgraph_checkpointer"
]
