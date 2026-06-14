"""Functional reliability helpers for Celery workers.

Celery workers run in a separate process from FastAPI, so they cannot use
`Request`-scoped dependencies such as `get_redis(request)`. Instead, the worker
should create one process-level Redis client using the same application factory
and pass that client into these functions.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from datetime import UTC, datetime
from inspect import isawaitable
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

type JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
type RedisOperationResult = object | Awaitable[object]


class RedisClientProtocol(Protocol):
    def set(self, *args: object, **kwargs: object) -> RedisOperationResult: ...
    def get(self, *args: object, **kwargs: object) -> RedisOperationResult: ...
    def delete(self, *args: object, **kwargs: object) -> RedisOperationResult: ...


type IdempotencyStatus = Literal["processing", "completed", "failed_permanent"]
type JsonMetadata = dict[str, "JsonValue"]

PROCESSING_STATUS: IdempotencyStatus = "processing"
COMPLETED_STATUS: IdempotencyStatus = "completed"
FAILED_PERMANENT_STATUS: IdempotencyStatus = "failed_permanent"

IDEMPOTENCY_NAMESPACE = "celery:idempotency"
CIRCUIT_BREAKER_NAMESPACE = "celery:circuit"


class IdempotencyRecord(BaseModel, frozen=True):
    """Serialized idempotency record stored in Redis."""

    status: IdempotencyStatus
    task_id: str | None = None
    updated_at: str
    metadata: JsonMetadata


class CircuitBreakerState(BaseModel, frozen=True):
    """Circuit breaker state snapshot."""

    state: str = "closed"
    failures: int = 0
    opened_at: float | None = None


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the circuit breaker is open."""


def run_redis_call[T](value: T | Awaitable[T]) -> T:
    """Resolve either a direct Redis result or an awaitable returned by async Redis."""
    if isawaitable(value):
        return asyncio.run(cast("Coroutine[object, object, T]", value))
    return value


def build_idempotency_key(
    idempotency_key: str,
    *,
    namespace: str = IDEMPOTENCY_NAMESPACE,
) -> str:
    return f"{namespace}:{idempotency_key}"


def serialize_idempotency_record(
    status: IdempotencyStatus,
    *,
    task_id: str | None = None,
    updated_at: str | None = None,
    metadata: JsonMetadata | None = None,
) -> str:
    record = IdempotencyRecord(
        status=status,
        task_id=task_id,
        updated_at=updated_at if updated_at is not None else datetime.now(tz=UTC).isoformat(),
        metadata=metadata or {},
    )
    return record.model_dump_json()


def acquire_idempotency_lock(
    redis_client: RedisClientProtocol,
    idempotency_key: str,
    *,
    task_id: str | None = None,
    ttl_seconds: int = 86400,
    metadata: JsonMetadata | None = None,
    namespace: str = IDEMPOTENCY_NAMESPACE,
) -> bool:
    """Acquire a processing lock for a business operation."""
    return bool(
        run_redis_call(
            redis_client.set(
                name=build_idempotency_key(idempotency_key, namespace=namespace),
                value=serialize_idempotency_record(
                    PROCESSING_STATUS,
                    task_id=task_id,
                    metadata=metadata,
                ),
                ex=ttl_seconds,
                nx=True,
            )
        )
    )


def mark_idempotency_completed(
    redis_client: RedisClientProtocol,
    idempotency_key: str,
    *,
    task_id: str | None = None,
    ttl_seconds: int = 86400,
    metadata: JsonMetadata | None = None,
    namespace: str = IDEMPOTENCY_NAMESPACE,
) -> None:
    run_redis_call(
        redis_client.set(
            name=build_idempotency_key(idempotency_key, namespace=namespace),
            value=serialize_idempotency_record(
                COMPLETED_STATUS,
                task_id=task_id,
                metadata=metadata,
            ),
            ex=ttl_seconds,
        )
    )


def mark_idempotency_failed_permanently(
    redis_client: RedisClientProtocol,
    idempotency_key: str,
    *,
    task_id: str | None = None,
    ttl_seconds: int = 86400,
    metadata: JsonMetadata | None = None,
    namespace: str = IDEMPOTENCY_NAMESPACE,
) -> None:
    run_redis_call(
        redis_client.set(
            name=build_idempotency_key(idempotency_key, namespace=namespace),
            value=serialize_idempotency_record(
                FAILED_PERMANENT_STATUS,
                task_id=task_id,
                metadata=metadata,
            ),
            ex=ttl_seconds,
        )
    )


def release_idempotency_processing_lock(
    redis_client: RedisClientProtocol,
    idempotency_key: str,
    *,
    namespace: str = IDEMPOTENCY_NAMESPACE,
) -> None:
    """Release the processing lock so a later retry can acquire it again."""
    run_redis_call(redis_client.delete(build_idempotency_key(idempotency_key, namespace=namespace)))


def get_idempotency_status(
    redis_client: RedisClientProtocol,
    idempotency_key: str,
    *,
    namespace: str = IDEMPOTENCY_NAMESPACE,
) -> IdempotencyStatus | None:
    payload = cast(
        "str | None",
        run_redis_call(
            redis_client.get(build_idempotency_key(idempotency_key, namespace=namespace))
        ),
    )
    if not payload:
        return None

    record = IdempotencyRecord.model_validate_json(payload)
    return record.status


def build_circuit_breaker_key(
    name: str,
    *,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> str:
    return f"{namespace}:{name}"


def get_circuit_breaker_state(
    redis_client: RedisClientProtocol,
    name: str,
    *,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> CircuitBreakerState:
    payload = cast(
        "str | None",
        run_redis_call(redis_client.get(build_circuit_breaker_key(name, namespace=namespace))),
    )
    if not payload:
        return CircuitBreakerState()
    return CircuitBreakerState.model_validate_json(payload)


def set_circuit_breaker_state(
    redis_client: RedisClientProtocol,
    name: str,
    state: CircuitBreakerState,
    *,
    recovery_timeout_seconds: int,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> None:
    run_redis_call(
        redis_client.set(
            name=build_circuit_breaker_key(name, namespace=namespace),
            value=state.model_dump_json(),
            ex=recovery_timeout_seconds * 2,
        )
    )


def is_circuit_breaker_open(
    redis_client: RedisClientProtocol,
    name: str,
    *,
    recovery_timeout_seconds: int,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> bool:
    state = get_circuit_breaker_state(redis_client, name, namespace=namespace)
    if state.state != "open" or state.opened_at is None:
        return False

    elapsed = time.time() - state.opened_at
    if elapsed < recovery_timeout_seconds:
        return True

    set_circuit_breaker_state(
        redis_client,
        name,
        CircuitBreakerState(state="half_open", failures=state.failures, opened_at=state.opened_at),
        recovery_timeout_seconds=recovery_timeout_seconds,
        namespace=namespace,
    )
    return False


def record_circuit_breaker_success(
    redis_client: RedisClientProtocol,
    name: str,
    *,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> None:
    run_redis_call(redis_client.delete(build_circuit_breaker_key(name, namespace=namespace)))


def record_circuit_breaker_failure(
    redis_client: RedisClientProtocol,
    name: str,
    *,
    failure_threshold: int,
    recovery_timeout_seconds: int,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> None:
    state = get_circuit_breaker_state(redis_client, name, namespace=namespace)
    failures = state.failures + 1

    if failures >= failure_threshold:
        set_circuit_breaker_state(
            redis_client,
            name,
            CircuitBreakerState(state="open", failures=failures, opened_at=time.time()),
            recovery_timeout_seconds=recovery_timeout_seconds,
            namespace=namespace,
        )
        return

    set_circuit_breaker_state(
        redis_client,
        name,
        CircuitBreakerState(state="closed", failures=failures),
        recovery_timeout_seconds=recovery_timeout_seconds,
        namespace=namespace,
    )


def run_with_circuit_breaker[T](
    redis_client: RedisClientProtocol,
    name: str,
    operation: Callable[[], T],
    *,
    failure_threshold: int,
    recovery_timeout_seconds: int,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> T:
    """Execute an operation unless the circuit breaker is open."""
    if is_circuit_breaker_open(
        redis_client,
        name,
        recovery_timeout_seconds=recovery_timeout_seconds,
        namespace=namespace,
    ):
        msg = f"Circuit breaker open for '{name}'"
        raise CircuitBreakerOpenError(msg)

    try:
        result = operation()
    except Exception:
        record_circuit_breaker_failure(
            redis_client,
            name,
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
            namespace=namespace,
        )
        raise

    record_circuit_breaker_success(redis_client, name, namespace=namespace)
    return result
