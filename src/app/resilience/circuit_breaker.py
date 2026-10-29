"""Standalone async Redis circuit-breaker example with an injectable store."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, override
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict
from redis.exceptions import RedisError

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

type AsyncOperation[T] = Callable[[], Awaitable[T]]


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AcquireStatus(StrEnum):
    ALLOW = "allow"
    ALLOW_PROBE = "allow_probe"
    REJECT = "reject"


class CircuitBreakerOpenError(RuntimeError):
    """Raised when the breaker is open and its recovery period has not elapsed."""


class CircuitBreakerBackendError(RuntimeError):
    """Raised when Redis is unavailable and the breaker is configured fail-closed."""


class CircuitBreakerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    probe_lease_seconds: int = 10
    fail_open: bool = True

    @override
    def model_post_init(self, __context: object) -> None:
        if self.failure_threshold < 1:
            msg = "failure_threshold must be at least 1"
            raise ValueError(msg)
        if self.recovery_timeout_seconds < 1:
            msg = "recovery_timeout_seconds must be at least 1"
            raise ValueError(msg)
        if self.probe_lease_seconds < 1:
            msg = "probe_lease_seconds must be at least 1"
            raise ValueError(msg)


class CircuitBreakerStore(Protocol):
    """Persistence boundary; the breaker core does not depend on Redis."""

    async def acquire(
        self,
        name: str,
        *,
        probe_lease_ms: int,
    ) -> AcquireStatus: ...

    async def record_success(self, name: str) -> None: ...

    async def record_failure(
        self,
        name: str,
        *,
        failure_threshold: int,
        recovery_timeout_ms: int,
    ) -> None: ...


class RedisCircuitBreakerStore:
    """Atomic distributed state transitions backed by an async redis-py client."""

    _ACQUIRE_SCRIPT = """
    local state_key = KEYS[1]
    local open_until_key = KEYS[2]
    local probe_key = KEYS[3]
    local redis_time = redis.call("TIME")
    local now_ms = redis_time[1] * 1000 + math.floor(redis_time[2] / 1000)
    local probe_lease_ms = tonumber(ARGV[1])
    local state = redis.call("GET", state_key) or "closed"

    if state == "closed" then
        return "allow"
    end

    if state == "open" then
        local open_until_ms = tonumber(redis.call("GET", open_until_key) or "0")
        if now_ms < open_until_ms then
            return "reject"
        end
        if redis.call("SET", probe_key, "1", "NX", "PX", probe_lease_ms) then
            redis.call("SET", state_key, "half_open")
            return "allow_probe"
        end
        return "reject"
    end

    if state == "half_open" then
        if redis.call("SET", probe_key, "1", "NX", "PX", probe_lease_ms) then
            return "allow_probe"
        end
        return "reject"
    end

    return redis.error_reply("unknown circuit-breaker state: " .. state)
    """

    _SUCCESS_SCRIPT = """
    redis.call("SET", KEYS[1], "closed")
    redis.call("DEL", KEYS[2], KEYS[3], KEYS[4])
    return 1
    """

    _FAILURE_SCRIPT = """
    local state_key = KEYS[1]
    local failures_key = KEYS[2]
    local open_until_key = KEYS[3]
    local probe_key = KEYS[4]
    local redis_time = redis.call("TIME")
    local now_ms = redis_time[1] * 1000 + math.floor(redis_time[2] / 1000)
    local failure_threshold = tonumber(ARGV[1])
    local recovery_timeout_ms = tonumber(ARGV[2])
    local state = redis.call("GET", state_key) or "closed"

    if state == "open" then
        return 1
    end

    if state == "half_open" then
        redis.call("SET", state_key, "open")
        redis.call("SET", open_until_key, now_ms + recovery_timeout_ms)
        redis.call("DEL", probe_key)
        return 1
    end

    local failures = redis.call("INCR", failures_key)
    if failures == 1 then
        redis.call("PEXPIRE", failures_key, recovery_timeout_ms * 2)
    end
    if failures >= failure_threshold then
        redis.call("SET", state_key, "open")
        redis.call("SET", open_until_key, now_ms + recovery_timeout_ms)
        redis.call("DEL", probe_key)
    end
    return 1
    """

    def __init__(self, redis_client: Redis, *, key_prefix: str = "examples:circuit") -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix.rstrip(":")
        self._acquire_script = redis_client.register_script(self._ACQUIRE_SCRIPT)
        self._success_script = redis_client.register_script(self._SUCCESS_SCRIPT)
        self._failure_script = redis_client.register_script(self._FAILURE_SCRIPT)

    def _keys(self, name: str) -> tuple[str, str, str, str]:
        breaker_key = f"{self._key_prefix}:{{{quote(name, safe='')}}}"
        return (
            f"{breaker_key}:state",
            f"{breaker_key}:open-until",
            f"{breaker_key}:probe",
            f"{breaker_key}:failures",
        )

    async def acquire(
        self,
        name: str,
        *,
        probe_lease_ms: int,
    ) -> AcquireStatus:
        state_key, open_until_key, probe_key, _ = self._keys(name)
        result = await self._acquire_script(
            keys=[state_key, open_until_key, probe_key],
            args=[probe_lease_ms],
        )
        status = result.decode() if isinstance(result, bytes) else str(result)
        return AcquireStatus(status)

    async def record_success(self, name: str) -> None:
        state_key, open_until_key, probe_key, failures_key = self._keys(name)
        await self._success_script(
            keys=[state_key, open_until_key, probe_key, failures_key],
            args=[],
        )

    async def record_failure(
        self,
        name: str,
        *,
        failure_threshold: int,
        recovery_timeout_ms: int,
    ) -> None:
        state_key, open_until_key, probe_key, failures_key = self._keys(name)
        await self._failure_script(
            keys=[state_key, failures_key, open_until_key, probe_key],
            args=[failure_threshold, recovery_timeout_ms],
        )


class CircuitBreakerService:
    """Async circuit breaker whose state store is supplied by the caller."""

    def __init__(
        self,
        store: CircuitBreakerStore,
        settings: CircuitBreakerSettings | None = None,
    ) -> None:
        self._store = store
        self._settings = settings or CircuitBreakerSettings()

    async def call[T](self, name: str, operation: AsyncOperation[T]) -> T:
        """Run one downstream operation and update its breaker state.

        Keep ``operation`` limited to the downstream call. Local parsing and
        application logic should run after this method so their errors do not
        count as downstream failures.
        """
        status = await self._acquire(name)
        if status is AcquireStatus.REJECT:
            msg = f"Circuit breaker is open for {name!r}"
            raise CircuitBreakerOpenError(msg)

        try:
            result = await operation()
        except Exception as operation_error:
            await self._record_failure(name, operation_error)
            raise

        await self._record_success(name)
        return result

    async def _acquire(self, name: str) -> AcquireStatus:
        try:
            return await self._store.acquire(
                name,
                probe_lease_ms=self._settings.probe_lease_seconds * 1000,
            )
        except RedisError as exc:
            if self._settings.fail_open:
                logger.warning(
                    "Circuit-breaker state unavailable; allowing request",
                    extra={"service": name},
                    exc_info=True,
                )
                return AcquireStatus.ALLOW
            msg = f"Circuit-breaker state unavailable for {name!r}"
            raise CircuitBreakerBackendError(msg) from exc

    async def _record_failure(self, name: str, operation_error: Exception) -> None:
        try:
            await self._store.record_failure(
                name,
                failure_threshold=self._settings.failure_threshold,
                recovery_timeout_ms=self._settings.recovery_timeout_seconds * 1000,
            )
        except RedisError as exc:
            logger.warning(
                "Circuit-breaker failure update failed",
                extra={"service": name},
                exc_info=True,
            )
            operation_error.add_note(f"circuit-breaker state update failed: {exc!r}")

    async def _record_success(self, name: str) -> None:
        try:
            await self._store.record_success(name)
        except RedisError as exc:
            logger.warning(
                "Circuit-breaker success update failed",
                extra={"service": name},
                exc_info=True,
            )
            if not self._settings.fail_open:
                msg = f"Circuit-breaker state update failed for {name!r}"
                raise CircuitBreakerBackendError(msg) from exc
