"""Redis health probes."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from redis import RedisError

from app.features.health.application.dto.result import DependencyHealth

from ....application.dto.result import ComponentCheck

if TYPE_CHECKING:
    from typing import Any

    from redis.asyncio import Redis

_DEEP_TIMEOUT_S = 2.0


class RedisProbe:
    """Probe Redis through the injected async client."""

    name = "redis"

    def __init__(self, client: Redis | None) -> None:
        self._client = client

    async def check(self) -> ComponentCheck:
        if self._client is None:
            return ComponentCheck.not_configured()
        try:
            start = time.perf_counter()
            await self._client.ping()
            response_time = (time.perf_counter() - start) * 1000
            info = await self._client.info()
            return ComponentCheck.healthy(
                response_time_ms=round(response_time, 2),
                version=info.get("redis_version", "unknown"),
                connected_clients=info.get("connected_clients", 0),
            )
        except RedisError as exc:
            exc.add_note("component=redis, operation=check_redis")
            return ComponentCheck.unhealthy(str(exc))


async def check_redis(app: Any) -> DependencyHealth:
    """Verify Redis connectivity for the root deep-health endpoint."""
    start = time.perf_counter()
    try:
        async with asyncio.timeout(_DEEP_TIMEOUT_S):
            await app.state.redis.ping()
    except (OSError, TimeoutError, RedisError) as exc:
        exc.add_note("dependency=redis, operation=deep_probe")
        return DependencyHealth.fail("redis", str(exc), (time.perf_counter() - start) * 1000)
    return DependencyHealth.ok("redis", (time.perf_counter() - start) * 1000)
