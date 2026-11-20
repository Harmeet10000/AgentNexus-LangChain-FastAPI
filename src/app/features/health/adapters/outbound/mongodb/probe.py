"""MongoDB health probes."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from pymongo.errors import PyMongoError

from app.features.health.application.dto.result import DependencyHealth

from ....application.dto.result import ComponentCheck

if TYPE_CHECKING:
    from typing import Any

    from motor.motor_asyncio import AsyncIOMotorClient

_DEEP_TIMEOUT_S = 2.0


class MongoDBProbe:
    """Probe MongoDB through the injected Motor client."""

    name = "mongodb"

    def __init__(self, client: AsyncIOMotorClient[Any] | None) -> None:
        self._client = client

    async def check(self) -> ComponentCheck:
        if self._client is None:
            return ComponentCheck.not_configured()
        try:
            start = time.perf_counter()
            await self._client.admin.command("ping")
            response_time = (time.perf_counter() - start) * 1000
            server_info = await self._client.server_info()
            return ComponentCheck.healthy(
                response_time_ms=round(response_time, 2),
                version=server_info.get("version", "unknown"),
            )
        except PyMongoError as exc:
            exc.add_note("component=mongodb, operation=check_mongodb")
            return ComponentCheck.unhealthy(str(exc))


async def check_mongodb(app: Any) -> DependencyHealth:
    """Verify MongoDB connectivity for the root deep-health endpoint."""
    start = time.perf_counter()
    try:
        async with asyncio.timeout(_DEEP_TIMEOUT_S):
            await app.state.mongo_client.admin.command("ping")
    except (OSError, TimeoutError, PyMongoError) as exc:
        exc.add_note("dependency=mongodb, operation=deep_probe")
        return DependencyHealth.fail("mongodb", str(exc), (time.perf_counter() - start) * 1000)
    return DependencyHealth.ok("mongodb", (time.perf_counter() - start) * 1000)
