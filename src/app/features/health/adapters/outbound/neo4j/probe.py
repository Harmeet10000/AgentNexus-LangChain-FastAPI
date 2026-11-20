"""Neo4j, Graphiti, Cognee, and plugin health probes."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from neo4j.exceptions import DriverError, Neo4jError

from app.features.health.application.dto.result import DependencyHealth, HealthStatus

from ....application.dto.result import AgentMemoryCheck, ComponentCheck

if TYPE_CHECKING:
    from typing import Any

    from neo4j import AsyncDriver

_DEEP_TIMEOUT_S = 2.0
_GRAPH_MEMORY_PROBE_TIMEOUT_S = 2.0
_GRAPH_MEMORY_PROBE_QUERY = "RETURN 1 AS ok"


class Neo4jProbe:
    """Run a read-only query against the application's Neo4j driver."""

    name = "neo4j"

    def __init__(self, driver: AsyncDriver | None) -> None:
        self._driver = driver

    async def check(self) -> ComponentCheck:
        if self._driver is None:
            return ComponentCheck.not_configured()
        try:
            start = time.perf_counter()
            async with self._driver.session() as session:
                result = await session.run(_GRAPH_MEMORY_PROBE_QUERY)
                record = await result.single()
            return ComponentCheck.healthy(
                response_time_ms=round((time.perf_counter() - start) * 1000, 2),
                ok=bool(record and record.get("ok") == 1),
            )
        except Neo4jError as exc:
            exc.add_note("component=neo4j, operation=check_neo4j")
            return ComponentCheck.unhealthy(str(exc))


class GraphitiProbe:
    """Probe the graph-memory client's read-only query surface."""

    name = "graphiti"

    def __init__(self, client: Any | None) -> None:
        self._client = client

    async def check(self) -> ComponentCheck:
        if self._client is None:
            return ComponentCheck.not_configured()
        try:
            start = time.perf_counter()
            async with asyncio.timeout(_GRAPH_MEMORY_PROBE_TIMEOUT_S):
                await self._client.driver.execute_query(_GRAPH_MEMORY_PROBE_QUERY)
        except (Neo4jError, DriverError, OSError, TimeoutError) as exc:
            exc.add_note("component=graphiti, operation=check_graphiti")
            # Driver errors can include a DSN; expose only the exception type.
            return ComponentCheck.unhealthy(type(exc).__name__)
        return ComponentCheck.healthy(
            response_time_ms=round((time.perf_counter() - start) * 1000, 2)
        )


class AgentMemoryProbe:
    """Report Cognee configuration and its Neo4j procedure precondition."""

    name = "agent_memory"

    def __init__(self, config: Any | None, driver: AsyncDriver | None) -> None:
        self._config = config
        self._driver = driver

    async def check(self) -> AgentMemoryCheck:
        if self._config is None:
            return AgentMemoryCheck(status=HealthStatus.DEGRADED, state="not_configured")

        graph_procedures_available = False
        graph_reachable = False
        if self._driver is not None:
            try:
                async with asyncio.timeout(_GRAPH_MEMORY_PROBE_TIMEOUT_S):
                    records, _, _ = await self._driver.execute_query(
                        "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.' "
                        "OR name STARTS WITH 'gds.' RETURN count(name) AS n"
                    )
                graph_reachable = True
                count = (
                    records[0].get("n", 0)
                    if records and hasattr(records[0], "get")
                    else getattr(records[0], "n", 0)
                    if records
                    else 0
                )
                graph_procedures_available = bool(records and count > 0)
            except (Neo4jError, DriverError, OSError, TimeoutError) as exc:
                exc.add_note("component=agent_memory, operation=check_agent_memory")
                return AgentMemoryCheck(
                    status=HealthStatus.UNHEALTHY,
                    state="disconnected",
                    error=type(exc).__name__,
                    graph_procedures_available=graph_procedures_available,
                )

        return AgentMemoryCheck(
            status=HealthStatus.HEALTHY,
            state="configured",
            graph_reachable=graph_reachable,
            graph_procedures_available=graph_procedures_available,
            embedding_dimension=self._config.embedding_dimension,
        )


async def check_neo4j(app: Any) -> DependencyHealth:
    """Verify Neo4j connectivity for the root deep-health endpoint."""
    start = time.perf_counter()
    driver = app.state.neo4j_driver
    if driver is None:
        return DependencyHealth.degraded("neo4j", "not initialised")
    try:
        async with asyncio.timeout(_DEEP_TIMEOUT_S):
            await driver.verify_connectivity()
    except (OSError, TimeoutError) as exc:
        exc.add_note("dependency=neo4j, operation=deep_probe")
        return DependencyHealth.fail("neo4j", str(exc), (time.perf_counter() - start) * 1000)
    return DependencyHealth.ok("neo4j", (time.perf_counter() - start) * 1000)


async def _procedure_count(session: Any, prefix: str) -> int:
    result = await session.run(
        "SHOW PROCEDURES YIELD name WHERE name STARTS WITH $prefix RETURN count(name) AS n",
        {"prefix": prefix},
    )
    record = await result.single()
    return int(record["n"]) if record is not None else 0


async def check_neo4j_plugins(app: Any) -> DependencyHealth:
    """Report APOC/GDS availability without hiding a bare Neo4j instance."""
    start = time.perf_counter()
    driver = getattr(app.state, "neo4j_driver", None)
    if driver is None:
        return DependencyHealth.degraded("neo4j-plugins", "not initialised")
    try:
        async with asyncio.timeout(_DEEP_TIMEOUT_S):
            async with driver.session() as session:
                apoc = await _procedure_count(session, "apoc.")
                gds = await _procedure_count(session, "gds.")
    except (OSError, TimeoutError, Neo4jError, DriverError) as exc:
        exc.add_note("dependency=neo4j-plugins, operation=deep_probe")
        return DependencyHealth.fail(
            "neo4j-plugins", str(exc), (time.perf_counter() - start) * 1000
        )
    latency = (time.perf_counter() - start) * 1000
    if apoc == 0:
        return DependencyHealth.fail(
            "neo4j-plugins",
            "APOC procedures not found; graph writes may fail with Cypher errors",
            latency,
        )
    if gds == 0:
        return DependencyHealth.degraded(
            "neo4j-plugins",
            "GDS procedures not found; only the optional graph-metrics helper "
            "calls GDS, never the add/cognify path",
            latency,
        )
    return DependencyHealth.ok("neo4j-plugins", latency)


async def check_graphiti(app: Any) -> DependencyHealth:
    """Verify Graphiti is initialised for the root deep-health endpoint."""
    start = time.perf_counter()
    if getattr(app.state, "graphiti", None) is None:
        return DependencyHealth.degraded("graphiti", "not initialised")
    return DependencyHealth.ok("graphiti", (time.perf_counter() - start) * 1000)


async def check_cognee(app: Any) -> DependencyHealth:
    """Probe Cognee's shared graph path with degraded, fail, and ok states."""
    start = time.perf_counter()
    config = getattr(app.state, "cognee_config", None)
    if config is None:
        return DependencyHealth.degraded("cognee", "not configured")
    driver = getattr(app.state, "neo4j_driver", None)
    if driver is None:
        return DependencyHealth.fail("cognee", "configured but no graph driver available")
    try:
        async with asyncio.timeout(_DEEP_TIMEOUT_S):
            await driver.verify_connectivity()
    except (OSError, TimeoutError) as exc:
        exc.add_note("dependency=cognee, operation=deep_probe")
        return DependencyHealth.fail("cognee", str(exc), (time.perf_counter() - start) * 1000)
    return DependencyHealth.ok("cognee", (time.perf_counter() - start) * 1000)
