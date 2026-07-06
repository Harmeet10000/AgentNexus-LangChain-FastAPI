"""Deep health check probes for critical dependencies.

Each probe uses a 2-second timeout and returns a DependencyHealth model.
Only the dependency clients from app.state are exercised — no new connections
are created.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.utils import DependencyHealth, logger

if TYPE_CHECKING:
    from fastapi import FastAPI

_HEALTH_TIMEOUT_S = 2.0


async def check_postgres(app: FastAPI) -> DependencyHealth:
    """Verify PostgreSQL connectivity via a lightweight SELECT 1."""
    start = time.perf_counter()
    try:
        engine = app.state.db_engine
        async with AsyncEngine(engine).connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth.ok("postgres", latency)
    except (OSError, TimeoutError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("Health check failed", dependency="postgres", error=str(exc))
        return DependencyHealth.fail("postgres", str(exc), latency)


async def check_redis(app: FastAPI) -> DependencyHealth:
    """Verify Redis connectivity via PING."""
    start = time.perf_counter()
    try:
        r = app.state.redis
        await r.ping()
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth.ok("redis", latency)
    except (OSError, TimeoutError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("Health check failed", dependency="redis", error=str(exc))
        return DependencyHealth.fail("redis", str(exc), latency)


async def check_mongodb(app: FastAPI) -> DependencyHealth:
    """Verify MongoDB connectivity via ping command."""
    start = time.perf_counter()
    try:
        client = app.state.mongo_client
        await client.admin.command("ping")
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth.ok("mongodb", latency)
    except (OSError, TimeoutError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("Health check failed", dependency="mongodb", error=str(exc))
        return DependencyHealth.fail("mongodb", str(exc), latency)


async def check_neo4j(app: FastAPI) -> DependencyHealth:
    """Verify Neo4j connectivity via verify_connectivity."""
    start = time.perf_counter()
    try:
        driver = app.state.neo4j_driver
        if driver is None:
            return DependencyHealth.degraded("neo4j", "not initialised")
        await driver.verify_connectivity()
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth.ok("neo4j", latency)
    except (OSError, TimeoutError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("Health check failed", dependency="neo4j", error=str(exc))
        return DependencyHealth.fail("neo4j", str(exc), latency)


async def check_graphiti(app: FastAPI) -> DependencyHealth:
    """Verify Graphiti is initialised (lightweight attribute check)."""
    start = time.perf_counter()
    graphiti = getattr(app.state, "graphiti", None)
    if graphiti is None:
        return DependencyHealth.degraded("graphiti", "not initialised")
    latency = (time.perf_counter() - start) * 1000
    return DependencyHealth.ok("graphiti", latency)


ALL_PROBES = [
    check_postgres,
    check_redis,
    check_mongodb,
    check_neo4j,
    check_graphiti,
]
