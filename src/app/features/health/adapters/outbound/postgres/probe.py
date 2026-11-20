"""PostgreSQL health probes."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.features.health.application.dto.result import DependencyHealth

from ....application.dto.result import ComponentCheck

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_DEEP_TIMEOUT_S = 2.0


class PostgresProbe:
    """Probe the application session factory with SELECT 1 and version."""

    name = "postgres"

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None) -> None:
        self._session_factory = session_factory

    async def check(self) -> ComponentCheck:
        if self._session_factory is None:
            return ComponentCheck.not_configured()
        start = time.perf_counter()
        try:
            async with self._session_factory() as session:
                await session.execute(select(1))
                version_result = await session.execute(select(func.version()))
                version = version_result.scalar() or "unknown"
        except SQLAlchemyError as exc:
            exc.add_note("table=health_probe, operation=check_postgres, query=SELECT 1")
            return ComponentCheck.unhealthy(str(exc))
        response_time = (time.perf_counter() - start) * 1000
        return ComponentCheck.healthy(
            response_time_ms=round(response_time, 2), version=str(version)
        )


async def check_postgres(app: Any) -> DependencyHealth:
    """Verify PostgreSQL connectivity for the root deep-health endpoint."""
    start = time.perf_counter()
    try:
        async with asyncio.timeout(_DEEP_TIMEOUT_S):
            engine = app.state.db_engine
            async with engine.connect() as connection:
                await connection.execute(select(1))
    except (OSError, TimeoutError, SQLAlchemyError) as exc:
        exc.add_note("dependency=postgres, operation=deep_probe")
        return DependencyHealth.fail("postgres", str(exc), (time.perf_counter() - start) * 1000)
    return DependencyHealth.ok("postgres", (time.perf_counter() - start) * 1000)
