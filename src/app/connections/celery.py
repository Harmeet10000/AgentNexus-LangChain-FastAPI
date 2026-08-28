"""Celery connection and production reliability configuration — single source.

Previously split across 4 files — now consolidated:
- `celery_task_names.py` remains the single definition site for task-name constants
  (imported here, so one copy serves the app, registry, and tests)
- `celery_reliability.py` → functional helpers (`Redis` from `app.utils.cache`,
  plain dicts, `RateLimitResult` as `BaseModel`) — now inlined and deleted
- `celery.py` → `ResilientTask`, `create_celery_app`, exchanges, routes, signals
- `celery_registry.py` → typed dispatch `CeleryTaskRegistry` + `TypedCeleryTask` — now inlined and deleted

All imports must use `from app.connections.celery import ...` or
`from app.connections.celery_task_names import ...` for constants.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import cache
from importlib import import_module
from inspect import isawaitable
from typing import TYPE_CHECKING, Literal, TypedDict, cast, override

import opentelemetry.trace as otel_trace
from celery import Celery, Task
from celery.exceptions import CeleryError
from celery.schedules import crontab
from celery.signals import (
    after_task_publish,
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
)
from kombu import Exchange, Queue
from opentelemetry import metrics
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.connections.redis import create_redis_client
from app.utils import logger
from app.utils.cache import Redis
from app.utils.json_serializer import from_json, to_json_str

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine
    from typing import Any, ClassVar

    from app.config.settings import Settings

# ---------------------------------------------------------------------------
# Task-name constants — imported from single definition site
# ---------------------------------------------------------------------------
from .celery_task_names import (
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_RECEIPT_GENERATION,
    BILLING_RECONCILIATION,
    CREDITS_EXPIRE,
    CREDITS_RECONCILE,
    INGESTION_TASK_NAMES,
    TASK_DECLARING_MODULES,
)

# ---------------------------------------------------------------------------
# Reliability helpers — functional, plain dicts, Redis from redis.asyncio
# ---------------------------------------------------------------------------

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


class IdempotencyLockError(RuntimeError):
    """Raised when idempotency lock cannot be acquired."""


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
    from loguru import logger  # noqa: PLC0415

    acquired = acquire_idempotency_lock(
        redis_client,
        idempotency_key,
        task_id=task_id,
        ttl_seconds=ttl_seconds,
        metadata=metadata,
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
        mark_idempotency_completed(
            redis_client,
            idempotency_key,
            task_id=task_id,
            ttl_seconds=ttl_seconds,
            metadata=metadata,
        )
        logger.info(
            "Operation completed successfully",
            idempotency_key=idempotency_key,
            task_id=task_id,
        )
    except Exception as exc:
        is_retryable = isinstance(exc, retryable_exceptions) if retryable_exceptions else False
        if is_retryable:
            release_idempotency_processing_lock(redis_client, idempotency_key)
            logger.warning(
                "Retryable failure, released processing lock",
                idempotency_key=idempotency_key,
                task_id=task_id,
                exception_type=type(exc).__name__,
            )
        else:
            mark_idempotency_failed_permanently(
                redis_client,
                idempotency_key,
                task_id=task_id,
                ttl_seconds=ttl_seconds,
                metadata=metadata,
            )
            logger.error(
                "Permanent failure, marked as failed",
                idempotency_key=idempotency_key,
                task_id=task_id,
                exception_type=type(exc).__name__,
            )
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
        from loguru import logger  # noqa: PLC0415

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
            await cast("Awaitable[object]", self._redis.zadd(key, {str(now): now}))
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
        from dataclasses import dataclass as _dc  # noqa: PLC0415

        @_dc(frozen=True)
        class _CBS:  # type: ignore[no-redef]
            state: str = "closed"
            failures: int = 0
            opened_at: float | None = None

            def model_dump_json(self) -> str:
                return to_json_str(
                    {"state": self.state, "failures": self.failures, "opened_at": self.opened_at}
                )

            @classmethod
            def model_validate_json(cls, payload: str) -> _CBS:
                d = cast("dict[str, object]", from_json(payload))
                failures_raw = d.get("failures", 0)
                return cls(
                    state=str(d.get("state", "closed")),
                    failures=failures_raw if isinstance(failures_raw, int) else 0,
                    opened_at=cast("float | None", d.get("opened_at")),
                )

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
        # ponytail: shim — old code did `from celery_reliability import RedisClientProtocol`; now use `Redis`
        return cast("type", Redis)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# ---------------------------------------------------------------------------
# Celery app — exchanges, routes, ResilientTask, factory, signals
# ---------------------------------------------------------------------------

settings = get_settings()

TASK_EXCHANGE = Exchange(
    name=settings.CELERY_DEFAULT_EXCHANGE,
    type="direct",
    durable=True,
)
TASK_DLX_EXCHANGE = Exchange(
    name=settings.CELERY_DEAD_LETTER_EXCHANGE,
    type="direct",
    durable=True,
)


def _task_routes() -> dict[str, dict[str, str]]:
    """Give every dispatchable name an explicit destination — no glob, no fallthrough."""
    default_route = {
        "queue": settings.CELERY_DEFAULT_QUEUE,
        "routing_key": settings.CELERY_DEFAULT_ROUTING_KEY,
    }
    ingestion_route = {
        "queue": settings.CELERY_INGESTION_QUEUE,
        "routing_key": settings.CELERY_INGESTION_ROUTING_KEY,
    }
    return {
        name: dict(ingestion_route if name in INGESTION_TASK_NAMES else default_route)
        for name in TASK_DECLARING_MODULES
    }


class ResilientTask(Task):
    """Base Celery task with retries, observability, and reliability helpers."""

    abstract = True
    autoretry_for: ClassVar[tuple[type[BaseException], ...]] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    retry_backoff: ClassVar[bool] = True
    retry_backoff_max: ClassVar[int] = settings.CELERY_RETRY_BACKOFF_MAX
    retry_jitter: ClassVar[bool] = True
    retry_kwargs: ClassVar[dict[str, int]] = {"max_retries": settings.CELERY_RETRY_MAX_RETRIES}

    _redis_client: ClassVar[Redis | None] = None

    @classmethod
    def get_redis_client(cls) -> Redis:
        if cls._redis_client is None:
            cls._redis_client = create_redis_client(settings.REDIS_URL)
        return cls._redis_client

    def acquire_idempotency_lock(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return acquire_idempotency_lock(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def mark_idempotency_completed(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        mark_idempotency_completed(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def mark_idempotency_failed_permanently(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        mark_idempotency_failed_permanently(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def release_idempotency_processing_lock(self, idempotency_key: str) -> None:
        release_idempotency_processing_lock(self.get_redis_client(), idempotency_key)

    def run_with_circuit_breaker(
        self,
        name: str,
        operation,
    ) -> Any:
        return run_with_circuit_breaker(
            self.get_redis_client(),
            name,
            operation,
            failure_threshold=settings.CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout_seconds=settings.CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        )

    @override
    def on_retry(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        _ = (args, kwargs, einfo)
        _celery_meters()[2].add(1, {"task_name": str(self.name)})
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).warning(f"Task scheduled for retry: {exc!s}")

    @override
    def on_failure(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        _ = (args, kwargs, einfo)
        completed, duration, retries = _celery_meters()
        attrs = {"task_name": str(self.name), "status": "failure"}
        completed.add(1, attrs)
        duration.record(time.time() - self.request.started or time.time(), attrs)
        retries.add(1, {"task_name": str(self.name)})
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).error(f"Task failed: {exc!s}")

    @override
    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
    ) -> None:
        _ = (retval, args, kwargs)
        completed, duration, _ = _celery_meters()
        attrs = {"task_name": str(self.name), "status": "success"}
        completed.add(1, attrs)
        duration.record(time.time() - self.request.started or time.time(), attrs)
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).info("Task completed successfully")


def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    app = Celery(
        main="langchain_fastapi",
        broker=settings.RABBITMQ_URL,
        backend="rpc://",
        include=[
            "tasks.agent_memory_tasks",
            "tasks.auth_email_tasks",
            "tasks.billing_tasks",
            "tasks.credit_tasks",
            "tasks.document_extraction_tasks",
            "tasks.document_tasks",
            "tasks.example",
            "tasks.pageindex_tasks",
        ],
    )
    app.Task = ResilientTask
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=None,
        broker_transport_options={"confirm_publish": True},
        task_publish_retry=True,
        task_publish_retry_policy={
            "max_retries": 3,
            "interval_start": 0.25,
            "interval_step": 0.5,
            "interval_max": 5,
        },
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=settings.CELERY_WORKER_MAX_TASKS_PER_CHILD,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_default_retry_delay=settings.CELERY_DEFAULT_RETRY_DELAY,
        task_track_started=True,
        task_send_sent_event=True,
        worker_send_task_events=True,
        task_default_delivery_mode="persistent",
        task_default_queue=settings.CELERY_DEFAULT_QUEUE,
        task_default_exchange=settings.CELERY_DEFAULT_EXCHANGE,
        task_default_exchange_type="direct",
        task_default_routing_key=settings.CELERY_DEFAULT_ROUTING_KEY,
        task_create_missing_queues=False,
        task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
        task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
        result_expires=settings.CELERY_TASK_RESULT_EXPIRES,
        task_queues=(
            Queue(
                name=settings.CELERY_DEFAULT_QUEUE,
                exchange=TASK_EXCHANGE,
                routing_key=settings.CELERY_DEFAULT_ROUTING_KEY,
                durable=True,
                queue_arguments={
                    "x-queue-type": "quorum",
                    "x-dead-letter-exchange": settings.CELERY_DEAD_LETTER_EXCHANGE,
                    "x-dead-letter-routing-key": settings.CELERY_DEAD_LETTER_ROUTING_KEY,
                },
            ),
            Queue(
                name=settings.CELERY_INGESTION_QUEUE,
                exchange=TASK_EXCHANGE,
                routing_key=settings.CELERY_INGESTION_ROUTING_KEY,
                durable=True,
                queue_arguments={
                    "x-queue-type": "quorum",
                    "x-dead-letter-exchange": settings.CELERY_DEAD_LETTER_EXCHANGE,
                    "x-dead-letter-routing-key": settings.CELERY_DEAD_LETTER_ROUTING_KEY,
                },
            ),
            Queue(
                name=settings.CELERY_DEAD_LETTER_QUEUE,
                exchange=TASK_DLX_EXCHANGE,
                routing_key=settings.CELERY_DEAD_LETTER_ROUTING_KEY,
                durable=True,
                queue_arguments={"x-queue-type": "quorum"},
            ),
        ),
        task_routes=_task_routes(),
        beat_schedule={
            "billing-invoice-daily": {
                "task": BILLING_INVOICE_GENERATION,
                "schedule": crontab(hour=0, minute=15),
            },
            "billing-dunning-daily": {
                "task": BILLING_DUNNING,
                "schedule": crontab(hour=1, minute=0),
            },
            "billing-receipt-daily": {
                "task": BILLING_RECEIPT_GENERATION,
                "schedule": crontab(hour=1, minute=45),
            },
            "billing-reconciliation-daily": {
                "task": BILLING_RECONCILIATION,
                "schedule": crontab(hour=2, minute=0),
            },
            "credits-expire-daily": {
                "task": CREDITS_EXPIRE,
                "schedule": crontab(
                    hour=settings.CREDIT_EXPIRATION_CRON_HOUR,
                    minute=settings.CREDIT_EXPIRATION_CRON_MINUTE,
                ),
            },
            "credits-reconcile-weekly": {
                "task": CREDITS_RECONCILE,
                "schedule": crontab(
                    hour=settings.CREDIT_RECONCILIATION_CRON_HOUR,
                    minute=settings.CREDIT_RECONCILIATION_CRON_MINUTE,
                    day_of_week=settings.CREDIT_RECONCILIATION_CRON_DAY_OF_WEEK,
                ),
            },
            "agent-memory-consolidation-nightly": {
                "task": "tasks.agent_memory_consolidation",
                "schedule": crontab(hour=3, minute=30),
                "args": ([],),
            },
        },
    )
    return app


celery_app = create_celery_app()


@cache
def _celery_meters() -> tuple[Any, Any, Any]:
    """Create the Celery meters on first use and reuse them for this process."""
    # Keep this lazy: importing the module must not initialize OTel exporters.
    meter = metrics.get_meter("celery")
    return (
        meter.create_counter("celery.task.completed_total", unit="1"),
        meter.create_histogram("celery.task.duration_seconds", unit="s"),
        meter.create_counter("celery.task.retries_total", unit="1"),
    )


@after_task_publish.connect
def log_task_published(
    sender: str | None = None,
    headers: dict[str, Any] | None = None,
    exchange: str | None = None,
    routing_key: str | None = None,
    **_: Any,
) -> None:
    logger.bind(
        task=sender,
        task_id=(headers or {}).get("id"),
        exchange=exchange,
        routing_key=routing_key,
    ).info("Celery task published")


@task_prerun.connect
def log_task_prerun(
    task_id: str | None = None,
    task: Task | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {
        "task": task.name if task else None,
        "task_id": task_id,
        "args_count": len(args or ()),
        "kwargs_keys": sorted((kwargs or {}).keys()),
    }
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).info("Celery task started")


@task_postrun.connect
def log_task_postrun(
    task_id: str | None = None,
    task: Task | None = None,
    state: str | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {"task": task.name if task else None, "task_id": task_id, "state": state}
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).info("Celery task finished")


@task_retry.connect
def log_task_retry(
    request: Any | None = None,
    reason: BaseException | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {
        "task": getattr(request, "task", None),
        "task_id": getattr(request, "id", None),
        "retry_count": getattr(request, "retries", None),
    }
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).warning(f"Celery task retry emitted: {reason!s}")


@task_failure.connect
def log_task_failure(
    task_id: str | None = None,
    exception: BaseException | None = None,
    sender: Task | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {"task": sender.name if sender else None, "task_id": task_id}
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).error(f"Celery task failed signal: {exception!s}")


# ---------------------------------------------------------------------------
# Typed registry — maps task names → Pydantic payload models for dispatch-time validation
# ---------------------------------------------------------------------------


class CeleryTaskPayload(BaseModel):
    """Base for all typed Celery task payloads."""

    model_config = {"extra": "forbid", "frozen": True}


class NoKwargsPayload(CeleryTaskPayload):
    """Payload for a task that takes no keyword arguments."""


class TaskDispatchError(CeleryError):
    """Base for the refusals the typed registry raises before a send."""


class UnregisteredTaskError(TaskDispatchError):
    """A dispatch named a task that has no registered payload model."""

    def __init__(self, task_name: str, *, known_names: frozenset[str]) -> None:
        self.task_name = task_name
        self.known_names = known_names
        message = (
            f"Celery task {task_name!r} has no registered payload model, so the dispatch was "
            f"refused rather than sent to a name no consumer may answer to. Register a "
            f"CeleryTaskPayload subclass for it in the module that declares it. "
            f"Registered names: {sorted(known_names)}"
        )
        super().__init__(message)


class TaskPayloadValidationError(TaskDispatchError):
    """A dispatched payload did not match the model its task declares."""

    def __init__(self, task_name: str, validation_error: ValidationError) -> None:
        self.task_name = task_name
        self.validation_error = validation_error
        message = (
            f"Payload for Celery task {task_name!r} does not match its registered model, so the "
            f"dispatch was refused rather than enqueued for a consumer that cannot accept it: "
            f"{validation_error}"
        )
        super().__init__(message)


class CeleryTaskRegistry:
    """Maps task names → Pydantic payload models for validation."""

    _registry: ClassVar[dict[str, type[CeleryTaskPayload]]] = {}

    @classmethod
    def register(cls, task_name: str, payload_model: type[CeleryTaskPayload]) -> None:
        cls._registry[task_name] = payload_model

    @classmethod
    def get(cls, task_name: str) -> type[CeleryTaskPayload] | None:
        return cls._registry.get(task_name)

    @classmethod
    def registered_names(cls) -> frozenset[str]:
        return frozenset(cls._registry)

    @classmethod
    def ensure_declared_module_imported(cls, task_name: str) -> None:
        if task_name in cls._registry:
            return
        module = TASK_DECLARING_MODULES.get(task_name)
        if module is None:
            return
        import_module(module)

    @classmethod
    def typed_send(
        cls, task_name: str, kwargs: dict[str, object], **send_task_opts: object
    ) -> object:
        cls.ensure_declared_module_imported(task_name)
        cls.validate(task_name, kwargs)
        return celery_app.send_task(task_name, kwargs=kwargs, **send_task_opts)

    @classmethod
    def validate(cls, task_name: str, kwargs: dict[str, Any]) -> CeleryTaskPayload:
        model = cls._registry.get(task_name)
        if model is None:
            logger.bind(task=task_name, registered=sorted(cls._registry)).error(
                "Task name is not registered"
            )
            raise UnregisteredTaskError(task_name, known_names=cls.registered_names())
        try:
            return model.model_validate(kwargs)
        except ValidationError as exc:
            logger.bind(task=task_name, errors=exc.errors()).error("Task payload validation failed")
            raise TaskPayloadValidationError(task_name, exc) from exc


class TypedCeleryTask(Task):
    """Base Celery task that validates kwargs against a registered Pydantic model."""

    abstract = True
    _validated_payload: CeleryTaskPayload | None = None

    @property
    def validated_payload(self) -> CeleryTaskPayload:
        if self._validated_payload is None:
            msg = "validated_payload accessed before task execution"
            raise RuntimeError(msg)
        return self._validated_payload

    @override
    def before_start(self, task_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        super().before_start(task_id, args, kwargs)
        task_name = self.name or ""
        self._validated_payload = CeleryTaskRegistry.validate(task_name, kwargs)

    @override
    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        super().on_success(retval, task_id, args, kwargs)
        if self._validated_payload is not None:
            logger.bind(
                task=self.name, task_id=task_id, payload_type=type(self._validated_payload).__name__
            ).info("Typed task completed")
