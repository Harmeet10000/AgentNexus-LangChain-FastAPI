import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import IntEnum

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.shared.circuit_breaker.lua_scripts import (
    CB_ACQUIRE_SCRIPT,
    CB_FAILURE_SCRIPT,
    CB_SUCCESS_SCRIPT,
)
from app.utils import logger
from app.utils.exceptions import ServiceUnavailableException


class AcquireStatus(IntEnum):
    ALLOW = 1
    ALLOW_PROBE = 2
    REJECT = 0


@dataclass(slots=True, frozen=True)
class CircuitBreakerSettings:
    service_name: str
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    probe_ttl_seconds: int = 10

    @property
    def recovery_timeout_ms(self) -> int:
        return self.recovery_timeout_seconds * 1000

    @property
    def probe_ttl_ms(self) -> int:
        return self.probe_ttl_seconds * 1000


@dataclass(slots=True, frozen=True)
class CircuitBreakerKeys:
    state: str
    failures: str
    timeout: str


class CircuitBreakerService:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis
        self._acquire_script = self.redis.register_script(CB_ACQUIRE_SCRIPT)
        self._success_script = self.redis.register_script(CB_SUCCESS_SCRIPT)
        self._failure_script = self.redis.register_script(CB_FAILURE_SCRIPT)

    @asynccontextmanager
    async def protect(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        probe_ttl_seconds: int = 10,
    ) -> AsyncIterator[None]:
        """Wrap a fragile downstream call with a Redis-backed circuit breaker."""
        settings = CircuitBreakerSettings(
            service_name=service_name,
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
            probe_ttl_seconds=probe_ttl_seconds,
        )
        keys = self._build_keys(service_name)

        status = await self._acquire_execution_slot(settings, keys)
        self._raise_if_rejected(settings, status)

        if status is AcquireStatus.ALLOW_PROBE:
            logger.bind(service=service_name).info(
                "Circuit breaker half-open; allowing a probe request."
            )

        try:
            yield
        except Exception:
            await self._record_failure(settings, keys)
            raise
        else:
            await self._record_success(service_name, keys)

    def _build_keys(self, service_name: str) -> CircuitBreakerKeys:
        return CircuitBreakerKeys(
            state=f"cb:state:{service_name}",
            failures=f"cb:failures:{service_name}",
            timeout=f"cb:timeout:{service_name}",
        )

    async def _acquire_execution_slot(
        self,
        settings: CircuitBreakerSettings,
        keys: CircuitBreakerKeys,
    ) -> AcquireStatus:
        try:
            raw_status = await self._acquire_script(
                keys=[keys.state, keys.timeout],
                args=[self._now_ms(), settings.probe_ttl_ms],
            )
            return AcquireStatus(raw_status)
        except RedisError as exc:
            logger.bind(service=settings.service_name, error=str(exc)).error(
                "Circuit breaker Redis acquire failed; allowing request."
            )
            return AcquireStatus.ALLOW

    def _raise_if_rejected(
        self,
        settings: CircuitBreakerSettings,
        status: AcquireStatus,
    ) -> None:
        if status is AcquireStatus.REJECT:
            logger.bind(service=settings.service_name).warning(
                "Circuit breaker open; rejecting request."
            )
            raise ServiceUnavailableException(
                detail=(f"Downstream service '{settings.service_name}' is currently unavailable.")
            )

    async def _record_success(
        self,
        service_name: str,
        keys: CircuitBreakerKeys,
    ) -> None:
        try:
            await self._success_script(keys=[keys.state, keys.failures])
            logger.bind(service=service_name).debug(
                "Circuit breaker recorded a successful downstream call."
            )
        except RedisError as exc:
            logger.bind(service=service_name, error=str(exc)).error(
                "Circuit breaker Redis success update failed."
            )

    async def _record_failure(
        self,
        settings: CircuitBreakerSettings,
        keys: CircuitBreakerKeys,
    ) -> None:
        try:
            await self._failure_script(
                keys=[keys.state, keys.failures, keys.timeout],
                args=[
                    self._now_ms(),
                    settings.failure_threshold,
                    settings.recovery_timeout_ms,
                ],
            )
            logger.bind(service=settings.service_name).warning(
                "Circuit breaker recorded a downstream failure."
            )
        except RedisError as exc:
            logger.bind(service=settings.service_name, error=str(exc)).error(
                "Circuit breaker Redis failure update failed."
            )

    def _now_ms(self) -> int:
        return int(time.time() * 1000)
