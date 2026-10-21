"""Worker reliability adapters. Redis leases are not exactly-once guarantees."""

from __future__ import annotations

import time
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from inspect import isawaitable
from typing import TYPE_CHECKING, Literal, TypedDict, cast
from uuid import uuid4

from pydantic import BaseModel

from app.observability.logging import logger
from app.utils.cache import Redis
from app.utils.json_serializer import from_json, to_json_str

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

    from app.config.settings import Settings


from app.shared.jobs.errors import CircuitBreakerOpenError, IdempotencyLockError
from app.worker.execution.runtime import run

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
type RedisOperationResult = object | Awaitable[object]

type IdempotencyStatus = Literal["processing", "completed", "failed_permanent"]
type JsonMetadata = dict[str, JsonValue]

PROCESSING_STATUS: IdempotencyStatus = "processing"
COMPLETED_STATUS: IdempotencyStatus = "completed"
FAILED_PERMANENT_STATUS: IdempotencyStatus = "failed_permanent"

IDEMPOTENCY_NAMESPACE = "celery:idempotency"
CIRCUIT_BREAKER_NAMESPACE = "celery:circuit"


class CircuitBreakerSnapshot(TypedDict):
    state: str
    failures: int
    opened_at: float | None


def run_redis_call[T](value: T | Awaitable[T]) -> T:
    """Resolve either a direct Redis result or an awaitable returned by async Redis."""
    if isawaitable(value):
        return run(cast("Coroutine[object, object, T]", value))
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
    payload: dict[str, object] = {
        "status": status,
        "task_id": task_id,
        "updated_at": updated_at if updated_at is not None else datetime.now(tz=UTC).isoformat(),
        "metadata": metadata or {},
    }
    return to_json_str(payload)


def acquire_idempotency_lock(
    redis_client: Redis,
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
    redis_client: Redis,
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
    redis_client: Redis,
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
    redis_client: Redis,
    idempotency_key: str,
    *,
    namespace: str = IDEMPOTENCY_NAMESPACE,
) -> None:
    """Release the processing lock so a later retry can acquire it again."""
    run_redis_call(redis_client.delete(build_idempotency_key(idempotency_key, namespace=namespace)))


def get_idempotency_status(
    redis_client: Redis,
    idempotency_key: str,
    *,
    namespace: str = IDEMPOTENCY_NAMESPACE,
) -> IdempotencyStatus | None:
    payload: str | None = cast(
        "str | None",
        run_redis_call(
            redis_client.get(build_idempotency_key(idempotency_key, namespace=namespace))
        ),
    )
    if not payload:
        return None
    data = cast("dict[str, object]", from_json(payload))
    status = data.get("status")
    if status in {PROCESSING_STATUS, COMPLETED_STATUS, FAILED_PERMANENT_STATUS}:
        return cast("IdempotencyStatus", status)
    return None


def build_circuit_breaker_key(
    name: str,
    *,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> str:
    return f"{namespace}:{name}"


def _default_circuit_snapshot() -> CircuitBreakerSnapshot:
    return {"state": "closed", "failures": 0, "opened_at": None}


def get_circuit_breaker_state(
    redis_client: Redis,
    name: str,
    *,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> CircuitBreakerSnapshot:
    payload = cast(
        "str | None",
        run_redis_call(redis_client.get(build_circuit_breaker_key(name, namespace=namespace))),
    )
    if not payload:
        return _default_circuit_snapshot()
    data = cast("dict[str, object]", from_json(payload))
    failures_raw = data.get("failures", 0)
    return {
        "state": str(data.get("state", "closed")),
        "failures": failures_raw if isinstance(failures_raw, int) else 0,
        "opened_at": cast("float | None", data.get("opened_at")),
    }


def set_circuit_breaker_state(
    redis_client: Redis,
    name: str,
    state: CircuitBreakerSnapshot,
    *,
    recovery_timeout_seconds: int,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> None:
    run_redis_call(
        redis_client.set(
            name=build_circuit_breaker_key(name, namespace=namespace),
            value=to_json_str(state),
            ex=recovery_timeout_seconds * 2,
        )
    )


def is_circuit_breaker_open(
    redis_client: Redis,
    name: str,
    *,
    recovery_timeout_seconds: int,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> bool:
    state = get_circuit_breaker_state(redis_client, name, namespace=namespace)
    if state["state"] != "open" or state["opened_at"] is None:
        return False
    elapsed = time.time() - state["opened_at"]
    if elapsed < recovery_timeout_seconds:
        return True
    set_circuit_breaker_state(
        redis_client,
        name,
        {"state": "half_open", "failures": state["failures"], "opened_at": state["opened_at"]},
        recovery_timeout_seconds=recovery_timeout_seconds,
        namespace=namespace,
    )
    return False


def record_circuit_breaker_success(
    redis_client: Redis,
    name: str,
    *,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> None:
    run_redis_call(redis_client.delete(build_circuit_breaker_key(name, namespace=namespace)))


def record_circuit_breaker_failure(
    redis_client: Redis,
    name: str,
    *,
    failure_threshold: int,
    recovery_timeout_seconds: int,
    namespace: str = CIRCUIT_BREAKER_NAMESPACE,
) -> None:
    state = get_circuit_breaker_state(redis_client, name, namespace=namespace)
    failures = state["failures"] + 1
    if failures >= failure_threshold:
        set_circuit_breaker_state(
            redis_client,
            name,
            {"state": "open", "failures": failures, "opened_at": time.time()},
            recovery_timeout_seconds=recovery_timeout_seconds,
            namespace=namespace,
        )
        return
    set_circuit_breaker_state(
        redis_client,
        name,
        {"state": "closed", "failures": failures, "opened_at": None},
        recovery_timeout_seconds=recovery_timeout_seconds,
        namespace=namespace,
    )


def run_with_circuit_breaker[T](
    redis_client: Redis,
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
    except Exception as exc:
        exc.add_note(f"circuit_breaker={name}")
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


class ReliabilitySystem:
    """Unified reliability base for Celery tasks."""

    def __init__(
        self,
        redis_client: Redis,
        *,
        circuit_breaker_name: str,
        failure_threshold: int | None = None,
        recovery_timeout_seconds: int | None = None,
        idempotency_ttl_seconds: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._redis = redis_client
        self._circuit_breaker_name = circuit_breaker_name
        if settings is None:
            from app.config.settings import get_settings  # noqa: PLC0415

            settings = get_settings()
        self._failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else settings.CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD
        )
        self._recovery_timeout = (
            recovery_timeout_seconds
            if recovery_timeout_seconds is not None
            else settings.CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        )
        self._default_idempotency_ttl = (
            idempotency_ttl_seconds
            if idempotency_ttl_seconds is not None
            else settings.CELERY_IDEMPOTENCY_TTL_SECONDS
        )
        self._validate_config()

    def _validate_config(self) -> None:
        if self._failure_threshold < 1:
            msg = f"failure_threshold must be >= 1, got {self._failure_threshold}"
            raise ValueError(msg)
        if self._recovery_timeout < 1:
            msg = f"recovery_timeout_seconds must be >= 1, got {self._recovery_timeout}"
            raise ValueError(msg)
        if self._default_idempotency_ttl < 1:
            msg = f"idempotency_ttl_seconds must be >= 1, got {self._default_idempotency_ttl}"
            raise ValueError(msg)

    def check_circuit_breaker(self) -> None:
        if is_circuit_breaker_open(
            self._redis,
            self._circuit_breaker_name,
            recovery_timeout_seconds=self._recovery_timeout,
        ):
            msg = f"Circuit breaker '{self._circuit_breaker_name}' is open"
            raise CircuitBreakerOpenError(msg)

    def record_success(self) -> None:
        record_circuit_breaker_success(self._redis, self._circuit_breaker_name)

    def record_failure(self) -> None:
        record_circuit_breaker_failure(
            self._redis,
            self._circuit_breaker_name,
            failure_threshold=self._failure_threshold,
            recovery_timeout_seconds=self._recovery_timeout,
        )

    def get_idempotency_status(self, idempotency_key: str) -> IdempotencyStatus | None:
        return get_idempotency_status(self._redis, idempotency_key)

    @property
    def default_idempotency_ttl(self) -> int:
        return self._default_idempotency_ttl


async def _resolve_redis_result[T](value: T | Awaitable[T]) -> T:
    """Await async clients in place, retaining support for synchronous test doubles."""
    if isawaitable(value):
        return cast("T", await value)
    return value


@asynccontextmanager
async def idempotency_manager(
    redis_client: Redis,
    idempotency_key: str,
    *,
    task_id: str | None = None,
    ttl_seconds: int = 86400,
    metadata: JsonMetadata | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (),
) -> AsyncIterator[None]:
    """Context manager for idempotency lock lifecycle."""
    acquired = await _resolve_redis_result(
        redis_client.set(
            name=build_idempotency_key(idempotency_key),
            value=serialize_idempotency_record(
                PROCESSING_STATUS, task_id=task_id, metadata=metadata
            ),
            ex=ttl_seconds,
            nx=True,
        )
    )
    if not acquired:
        logger.warning(
            "Idempotency lock already held",
            idempotency_key=idempotency_key,
            task_id=task_id,
        )
        msg = f"Operation '{idempotency_key}' is already processing"
        raise IdempotencyLockError(msg)
    logger.info(
        "Idempotency lock acquired",
        idempotency_key=idempotency_key,
        task_id=task_id,
        ttl_seconds=ttl_seconds,
    )
    try:
        yield
        await _resolve_redis_result(
            redis_client.set(
                name=build_idempotency_key(idempotency_key),
                value=serialize_idempotency_record(
                    COMPLETED_STATUS, task_id=task_id, metadata=metadata
                ),
                ex=ttl_seconds,
            )
        )
        logger.info(
            "Operation completed successfully",
            idempotency_key=idempotency_key,
            task_id=task_id,
        )
    except Exception as exc:
        exc.add_note(f"idempotency_key={idempotency_key}, operation=idempotency_manager")
        is_retryable = isinstance(exc, retryable_exceptions) if retryable_exceptions else False
        if is_retryable:
            await _resolve_redis_result(redis_client.delete(build_idempotency_key(idempotency_key)))
            logger.warning(
                "Retryable failure, released processing lock",
                idempotency_key=idempotency_key,
                task_id=task_id,
                exception_type=type(exc).__name__,
            )
        else:
            await _resolve_redis_result(
                redis_client.set(
                    name=build_idempotency_key(idempotency_key),
                    value=serialize_idempotency_record(
                        FAILED_PERMANENT_STATUS, task_id=task_id, metadata=metadata
                    ),
                    ex=ttl_seconds,
                )
            )
            logger.bind(
                idempotency_key=idempotency_key,
                task_id=task_id,
                exception_type=type(exc).__name__,
                error=str(exc),
            ).exception("Permanent failure, marked as failed")
        raise


class RateLimitResult(BaseModel):
    """Result of rate limit check."""

    model_config = {"frozen": True}

    allowed: bool
    remaining: int
    reset_at: float
    scope: str


class RateLimiter:
    """Redis-based rate limiter with config embedded in keys."""

    def __init__(
        self,
        redis_client: Redis,
        *,
        scope: str,
        rate: int,
        period_seconds: int,
        burst: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._redis = redis_client
        self._scope = scope
        self._rate = rate
        self._period = period_seconds
        self._burst = burst if burst is not None else rate
        if settings is None:
            from app.config.settings import get_settings  # noqa: PLC0415

            settings = get_settings()
        self._trusted_proxies = settings.FASTAPI_GUARD_TRUSTED_PROXIES
        self._proxy_depth = settings.FASTAPI_GUARD_TRUSTED_PROXY_DEPTH
        self._validate_config()

    def _validate_config(self) -> None:
        if self._rate < 1:
            msg = f"rate must be >= 1, got {self._rate}"
            raise ValueError(msg)
        if self._period < 1:
            msg = f"period_seconds must be >= 1, got {self._period}"
            raise ValueError(msg)
        if self._burst < self._rate:
            msg = f"burst must be >= rate, got burst={self._burst}, rate={self._rate}"
            raise ValueError(msg)

    def _build_key(self) -> str:
        return (
            f"celery:ratelimit:{self._scope}"
            f":rate={self._rate}"
            f":period={self._period}"
            f":burst={self._burst}"
        )

    @staticmethod
    def _parse_key(key: str) -> dict[str, int | str]:
        parts = key.split(":")
        config_start = next(
            (i for i, p in enumerate(parts[2:], start=2) if "=" in p),
            len(parts),
        )
        scope = ":".join(parts[2:config_start])
        config_parts: dict[str, int] = {}
        for part in parts[config_start:]:
            if "=" in part:
                key_part, value = part.split("=", 1)
                config_parts[key_part] = int(value)
        return {
            "scope": scope,
            "rate": config_parts.get("rate", 0),
            "period": config_parts.get("period", 0),
            "burst": config_parts.get("burst", 0),
        }

    def extract_client_ip(self, forwarded_for: str | None, direct_ip: str) -> str:
        if not self._trusted_proxies or not forwarded_for:
            return direct_ip
        ip_chain = [ip.strip() for ip in forwarded_for.split(",")]
        if len(ip_chain) >= self._proxy_depth:
            return ip_chain[-(self._proxy_depth)]
        return ip_chain[0] if ip_chain else direct_ip

    async def check_and_increment(
        self,
        *,
        forwarded_for: str | None = None,
        direct_ip: str | None = None,
    ) -> RateLimitResult:
        now = time.time()
        key = self._build_key()
        final_scope = self._scope
        if direct_ip:
            client_ip = self.extract_client_ip(forwarded_for, direct_ip)
            final_scope = f"{self._scope}:ip={client_ip}"
            key = f"{key}:ip={client_ip}"
        window_start = now - self._period
        await cast("Awaitable[object]", self._redis.zremrangebyscore(key, "-inf", window_start))
        current_count = cast("int", await cast("Awaitable[object]", self._redis.zcard(key)))
        allowed = current_count < self._burst
        if allowed:
            member = f"{now:.9f}:{uuid4().hex}"
            await cast("Awaitable[object]", self._redis.zadd(key, {member: now}))
            await cast("Awaitable[object]", self._redis.expire(key, self._period * 2))
            remaining = self._burst - current_count - 1
        else:
            remaining = 0
        reset_at = now + self._period
        logger.debug(
            "Rate limit check",
            scope=final_scope,
            allowed=allowed,
            current=current_count,
            limit=self._burst,
            remaining=remaining,
        )
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            scope=final_scope,
        )


# Back-compat shims for removed datamodels — keep importable for one release
def __getattr__(name: str) -> type:
    if name == "CircuitBreakerState":
        from pydantic import BaseModel as _BM  # noqa: PLC0415, N814

        class _CBS(_BM, frozen=True):  # type: ignore[no-redef]
            state: str = "closed"
            failures: int = 0
            opened_at: float | None = None

        return _CBS  # type: ignore[return-value]
    if name == "IdempotencyRecord":
        from pydantic import BaseModel as _BM  # noqa: PLC0415, N814

        class _IR(_BM, frozen=True):  # type: ignore[no-redef]
            status: IdempotencyStatus
            task_id: str | None = None
            updated_at: str
            metadata: JsonMetadata

        return _IR  # type: ignore[return-value]
    if name == "RedisClientProtocol":
        # Back-compat shim: older imports expected a redis client alias; current runtime uses `Redis`.
        return cast("type", Redis)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
