"""Deprecated shim — use `app.connections.celery` instead.

`app/connections/celery.py` is now the single source for all Celery-related code
(constants → reliability helpers → app → registry). This file re-exports the
reliability helpers so `from app.connections.celery_reliability import ...` keeps
working for one release. New code must import from `app.connections.celery`.

Wire format and Redis typing are unchanged; `Redis` is imported from
`redis.asyncio` (same as `app.utils.cache`) and `RateLimitResult` is a
`BaseModel` per the clarification.
"""

import contextlib

from app.connections.celery import (
    CIRCUIT_BREAKER_NAMESPACE,
    COMPLETED_STATUS,
    FAILED_PERMANENT_STATUS,
    IDEMPOTENCY_NAMESPACE,
    PROCESSING_STATUS,
    CircuitBreakerOpenError,
    CircuitBreakerSnapshot,
    IdempotencyLockError,
    IdempotencyStatus,
    JsonMetadata,
    JsonValue,
    RateLimiter,
    RateLimitResult,
    RedisOperationResult,
    ReliabilitySystem,
    acquire_idempotency_lock,
    build_circuit_breaker_key,
    build_idempotency_key,
    get_circuit_breaker_state,
    get_idempotency_status,
    idempotency_manager,
    is_circuit_breaker_open,
    mark_idempotency_completed,
    mark_idempotency_failed_permanently,
    record_circuit_breaker_failure,
    record_circuit_breaker_success,
    release_idempotency_processing_lock,
    run_redis_call,
    run_with_circuit_breaker,
    serialize_idempotency_record,
    set_circuit_breaker_state,
)

# Back-compat for removed datamodels — still importable via `__getattr__` in the
# canonical module. Re-export explicitly so `from .celery_reliability import CircuitBreakerState`
# works without triggering `__getattr__` indirection in linters.
with contextlib.suppress(ImportError):
    from app.connections.celery import (
        CircuitBreakerState,
        IdempotencyRecord,
        RedisClientProtocol,
    )

__all__ = [
    "CIRCUIT_BREAKER_NAMESPACE",
    "COMPLETED_STATUS",
    "FAILED_PERMANENT_STATUS",
    "IDEMPOTENCY_NAMESPACE",
    "PROCESSING_STATUS",
    "CircuitBreakerOpenError",
    "CircuitBreakerSnapshot",
    "CircuitBreakerState",
    "IdempotencyLockError",
    "IdempotencyRecord",
    "IdempotencyStatus",
    "JsonMetadata",
    "JsonValue",
    "RateLimitResult",
    "RateLimiter",
    "RedisClientProtocol",
    "RedisOperationResult",
    "ReliabilitySystem",
    "acquire_idempotency_lock",
    "build_circuit_breaker_key",
    "build_idempotency_key",
    "get_circuit_breaker_state",
    "get_idempotency_status",
    "idempotency_manager",
    "is_circuit_breaker_open",
    "mark_idempotency_completed",
    "mark_idempotency_failed_permanently",
    "record_circuit_breaker_failure",
    "record_circuit_breaker_success",
    "release_idempotency_processing_lock",
    "run_redis_call",
    "run_with_circuit_breaker",
    "serialize_idempotency_record",
    "set_circuit_breaker_state",
]
