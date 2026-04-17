"""Database connection dependencies."""

from .celery import ResilientTask, celery_app, create_celery_app
from .celery_reliability import (
    CircuitBreakerOpenError,
    acquire_idempotency_lock,
    build_circuit_breaker_key,
    build_idempotency_key,
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
from .httpx_client import (
    create_httpx_client,
    get_httpx_client,
    get_shared_httpx_client,
)
from .mongodb import create_mongo_client, get_mongodb
from .neo4j import (
    close_neo4j_driver,
    get_neo4j_driver,
    get_neo4j_session,
    init_neo4j,
)
from .postgres import get_postgres_db, init_db
from .redis import create_redis_client, get_redis
from .tavily import (
    close_tavily_http_client,
    create_tavily_http_client,
    get_shared_tavily_http_client,
    get_tavily_http_client,
)

__all__ = [
    "ResilientTaskcelery_app",
    "close_neo4j_driver",
    "close_tavily_http_client",
    "create_celery_app",
    "create_httpx_client",
    "create_mongo_client",
    "create_redis_client",
    "create_tavily_http_client",
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
]
