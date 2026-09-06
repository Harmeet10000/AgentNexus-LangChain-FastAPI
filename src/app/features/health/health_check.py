"""Deep health check probes for critical dependencies."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ...utils import DependencyHealth, logger

if TYPE_CHECKING:
    from fastapi import FastAPI

_HEALTH_TIMEOUT_S = 2.0


async def check_postgres(app: FastAPI) -> DependencyHealth:
    """Verify PostgreSQL connectivity via a lightweight SELECT 1."""
    start = time.perf_counter()
    try:
        engine = app.state.db_engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth.ok("postgres", latency)
    except (OSError, TimeoutError, SQLAlchemyError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.bind(dependency="postgres", error=str(exc)).exception("Health check failed")
        return DependencyHealth.fail("postgres", str(exc), latency)


async def check_redis(app: FastAPI) -> DependencyHealth:
    """Verify Redis connectivity via PING."""
    start = time.perf_counter()
    try:
        redis = app.state.redis
        await redis.ping()
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth.ok("redis", latency)
    except (OSError, TimeoutError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.bind(dependency="redis", error=str(exc)).exception("Health check failed")
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
        logger.bind(dependency="mongodb", error=str(exc)).exception("Health check failed")
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
        logger.bind(dependency="neo4j", error=str(exc)).exception("Health check failed")
        return DependencyHealth.fail("neo4j", str(exc), latency)


async def check_graphiti(app: FastAPI) -> DependencyHealth:
    """Verify Graphiti is initialised."""
    start = time.perf_counter()
    if getattr(app.state, "graphiti", None) is None:
        return DependencyHealth.degraded("graphiti", "not initialised")
    latency = (time.perf_counter() - start) * 1000
    return DependencyHealth.ok("graphiti", latency)


async def check_cognee(app: FastAPI) -> DependencyHealth:
    """Probe Cognee's shared graph path with degraded, fail, and ok states."""
    start = time.perf_counter()
    config = getattr(app.state, "cognee_config", None)
    if config is None:
        return DependencyHealth.degraded("cognee", "not configured")

    driver = getattr(app.state, "neo4j_driver", None)
    if driver is None:
        return DependencyHealth.fail("cognee", "configured but no graph driver available")

    try:
        async with asyncio.timeout(_HEALTH_TIMEOUT_S):
            await driver.verify_connectivity()
    except (OSError, TimeoutError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.bind(dependency="cognee", error=str(exc)).exception("Health check failed")
        return DependencyHealth.fail("cognee", str(exc), latency)
    latency = (time.perf_counter() - start) * 1000
    return DependencyHealth.ok("cognee", latency)


ALL_PROBES = [
    check_postgres,
    check_redis,
    check_mongodb,
    check_neo4j,
    check_graphiti,
    check_cognee,
]
