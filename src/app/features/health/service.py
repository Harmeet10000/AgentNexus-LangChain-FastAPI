"""Health service layer."""

import asyncio
import os
import platform
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

import psutil
from celery import Celery
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import AsyncDriver
from neo4j.exceptions import DriverError, Neo4jError
from pymongo.errors import PyMongoError
from redis import RedisError
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.connections.cognee import CogneeSetupConfig
from app.shared.result import HealthStatus
from app.utils import logger, trace_layer

from .dto import (
    AgentMemoryCheck,
    ComponentCheck,
    HealthChecksDTO,
    HealthDataDTO,
    HealthResultDTO,
    SelfInfoDTO,
)

type Probe = Callable[[], Awaitable[ComponentCheck]]

type HealthProbeResults = tuple[
    ComponentCheck,
    ComponentCheck,
    ComponentCheck,
    ComponentCheck,
    ComponentCheck,
    AgentMemoryCheck,
    ComponentCheck,
    ComponentCheck,
    ComponentCheck,
    dict[str, Any],
    dict[str, Any],
]

# The graph-memory probe is bounded: an unreachable graph backend must report,
# not hang. A readiness probe that blocks is worse than one that answers degraded.
_GRAPH_MEMORY_PROBE_TIMEOUT_S = 2.0
_HEALTH_CHECK_TIMEOUT_S = 3.0

# Read-only liveness query. The graph-memory client is *set once at startup* and
# never cleared, so a presence check alone cannot distinguish "initialised" from
# "reachable" — the probe has to ask the backend something.
_GRAPH_MEMORY_PROBE_QUERY = "RETURN 1 AS ok"

# Shared default for absent optionals and empty registries — frozen, so safe
# to reuse across checks without copying.
_NULL_CHECK = ComponentCheck.not_configured()


async def _null_probe() -> ComponentCheck:
    """Null Object for an absent async dependency: answers, never fails."""
    return _NULL_CHECK


def _null_probe_sync() -> ComponentCheck:
    """Null Object for an absent sync dependency: answers, never fails."""
    return _NULL_CHECK


def _require[T](client: T | None, component: str) -> T:
    """Narrow an injected client; absent means the probe selector mis-wired.

    Probe selection guarantees a configured client before a check runs, so
    reaching here with ``None`` is a programming error that must fail fast
    to the GEH — never silently report healthy.
    """
    if client is None:
        message = f"unconfigured {component} client reached its check"
        raise AssertionError(message)
    return client


class GraphQueryDriver(Protocol):
    """Minimal query surface the graph-memory probe needs from its driver."""

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any: ...


class GraphMemoryClient(Protocol):
    """Structural type of the graph-memory client published on ``app.state.graphiti``.

    Declared structurally rather than imported: pulling ``graphiti_core`` in for a
    type would add ~1.7s to every import of the health feature, and the probe
    needs exactly one attribute.
    """

    driver: GraphQueryDriver


class HealthService:
    """Service for system and dependency health checks."""

    def __init__(
        self,
        mongo_client: AsyncIOMotorClient[Any] | None,
        redis_client: Redis | None,
        postgres_session_factory: async_sessionmaker[AsyncSession] | None,
        neo4j_driver: AsyncDriver | None,
        celery_app: Celery | None,
        *,
        graph_memory_client: GraphMemoryClient | None = None,
        cognee_config: CogneeSetupConfig | None = None,
    ) -> None:
        self.mongo_client = mongo_client
        self.redis_client = redis_client
        self.postgres_session_factory = postgres_session_factory
        self.neo4j_driver = neo4j_driver
        self.celery_app = celery_app
        self.graph_memory_client = graph_memory_client
        self.cognee_config = cognee_config
        self.start_time = time.time()

    @staticmethod
    @trace_layer("service")
    async def get_self_info(
        server_name: str,
        server_version: str,
        client_host: str,
    ) -> SelfInfoDTO:
        """Return basic service metadata."""
        return SelfInfoDTO(
            server=server_name,
            version=server_version,
            client=client_host,
            timestamp=time.time(),
        )

    @staticmethod
    def _wired(check: Probe, client: object | None) -> Probe:
        """Select the real probe when its client is configured, else the null probe."""
        return check if client is not None else _null_probe

    @trace_layer("service")
    async def get_health(self) -> HealthResultDTO:
        """Run all health checks and return aggregated status."""
        # Absent clients resolve to the null probe here, once — check methods
        # below therefore assume a configured client and never branch on None.
        # Health checks are independent. Running them concurrently keeps a slow
        # database from serialising the latency of every other dependency.
        # The cast pins each gather position to its probe type — heterogeneous
        # gather otherwise joins every element into a union.
        (
            database_check,
            redis_check,
            postgres_check,
            neo4j_check,
            graphiti_check,
            agent_memory_check,
            celery_check,
            memory_check,
            disk_check,
            system_health,
            application_health,
        ) = cast(
            "HealthProbeResults",
            await asyncio.gather(
                self._run_async_check(
                    self._wired(self._check_mongodb, self.mongo_client), "mongodb"
                ),
                self._run_async_check(self._wired(self._check_redis, self.redis_client), "redis"),
                self._run_async_check(
                    self._wired(self._check_postgres, self.postgres_session_factory), "postgres"
                ),
                self._run_async_check(self._wired(self._check_neo4j, self.neo4j_driver), "neo4j"),
                self._run_async_check(
                    self._wired(self._check_graphiti, self.graph_memory_client), "graphiti"
                ),
                self._guarded_agent_memory(),
                self._run_sync_check(
                    self._check_celery if self.celery_app is not None else _null_probe_sync,
                    "celery",
                ),
                self._run_sync_check(self._check_memory, "memory"),
                self._run_sync_check(self._check_disk, "disk"),
                asyncio.to_thread(self._get_system_health),
                asyncio.to_thread(self._get_application_health),
            ),
        )

        checks = HealthChecksDTO(
            database=database_check,
            redis=redis_check,
            postgres=postgres_check,
            neo4j=neo4j_check,
            graphiti=graphiti_check,
            celery=celery_check,
            memory=memory_check,
            disk=disk_check,
            agent_memory=agent_memory_check,
        )

        overall_status = self._compute_overall_status(checks=checks)
        status_code = 200 if overall_status == HealthStatus.HEALTHY else 503

        data = HealthDataDTO(
            status=overall_status,
            timestamp=time.time(),
            application=application_health,
            system=system_health,
            checks=checks,
        )

        logger.bind(status=overall_status, status_code=status_code).info("Health check evaluated")
        return HealthResultDTO(
            message=f"Health check: {overall_status}",
            status_code=status_code,
            data=data,
        )

    @staticmethod
    async def _run_async_check(check: Probe, component: str) -> ComponentCheck:
        try:
            async with asyncio.timeout(_HEALTH_CHECK_TIMEOUT_S):
                return await check()
        except TimeoutError as exc:
            exc.add_note(f"component={component}, operation=health_check")
            logger.bind(component=component, timeout_seconds=_HEALTH_CHECK_TIMEOUT_S).warning(
                "Health check timed out"
            )
            return ComponentCheck.timeout(component, _HEALTH_CHECK_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — health endpoint must fail closed
            exc.add_note(f"component={component}, operation=health_check")
            logger.bind(component=component, error_type=type(exc).__name__).exception(
                "Health check failed unexpectedly"
            )
            return ComponentCheck.unhealthy(type(exc).__name__, state="error")

    @staticmethod
    async def _run_sync_check(
        check: Callable[[], ComponentCheck], component: str
    ) -> ComponentCheck:
        try:
            return await asyncio.wait_for(asyncio.to_thread(check), timeout=_HEALTH_CHECK_TIMEOUT_S)
        except TimeoutError as exc:
            exc.add_note(f"component={component}, operation=health_check")
            logger.bind(component=component, timeout_seconds=_HEALTH_CHECK_TIMEOUT_S).warning(
                "Health check timed out"
            )
            return ComponentCheck.timeout(component, _HEALTH_CHECK_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — health endpoint must fail closed
            exc.add_note(f"component={component}, operation=health_check")
            logger.bind(component=component, error_type=type(exc).__name__).exception(
                "Health check failed unexpectedly"
            )
            return ComponentCheck.unhealthy(type(exc).__name__, state="error")

    async def _check_mongodb(self) -> ComponentCheck:
        client = self.mongo_client
        client = _require(client, "mongodb")
        try:
            start = time.perf_counter()
            await client.admin.command("ping")
            response_time = (time.perf_counter() - start) * 1000
            server_info = await client.server_info()
            return ComponentCheck.healthy(
                response_time_ms=round(response_time, 2),
                version=server_info.get("version", "unknown"),
            )
        except PyMongoError as exc:
            exc.add_note("component=mongodb, operation=check_mongodb")
            logger.bind(error=str(exc), component="mongodb").exception(
                "MongoDB health check failed"
            )
            return ComponentCheck.unhealthy(str(exc))

    async def _check_redis(self) -> ComponentCheck:
        redis_client = self.redis_client
        redis_client = _require(redis_client, "redis")
        try:
            start = time.perf_counter()
            await redis_client.ping()
            response_time = (time.perf_counter() - start) * 1000
            info = await redis_client.info()
            return ComponentCheck.healthy(
                response_time_ms=round(response_time, 2),
                version=info.get("redis_version", "unknown"),
                connected_clients=info.get("connected_clients", 0),
            )
        except RedisError as exc:
            exc.add_note("component=redis, operation=check_redis")
            logger.bind(error=str(exc), component="redis").exception("Redis health check failed")
            return ComponentCheck.unhealthy(str(exc))

    async def _check_postgres(self) -> ComponentCheck:
        session_factory = self.postgres_session_factory
        session_factory = _require(session_factory, "postgres")
        start = time.perf_counter()
        try:
            async with session_factory() as session:
                await session.execute(text("SELECT 1"))
                version_result = await session.execute(text("SELECT version()"))
                version = version_result.scalar() or "unknown"
        except SQLAlchemyError as exc:
            exc.add_note("table=health_probe, operation=check_postgres, query=SELECT 1")
            logger.bind(
                error=str(exc),
                component="postgres",
                table="health_probe",
                operation="check_postgres",
                query="SELECT 1",
            ).exception("Postgres health check failed")
            return ComponentCheck.unhealthy(str(exc))
        response_time = (time.perf_counter() - start) * 1000
        return ComponentCheck.healthy(
            response_time_ms=round(response_time, 2), version=str(version)
        )

    async def _check_neo4j(self) -> ComponentCheck:
        driver = self.neo4j_driver
        driver = _require(driver, "neo4j")
        try:
            start = time.perf_counter()
            async with driver.session() as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
            response_time = (time.perf_counter() - start) * 1000
            return ComponentCheck.healthy(
                response_time_ms=round(response_time, 2),
                ok=bool(record and record.get("ok") == 1),
            )
        except Neo4jError as exc:
            exc.add_note("component=neo4j, operation=check_neo4j")
            logger.bind(error=str(exc), component="neo4j").exception("Neo4j health check failed")
            return ComponentCheck.unhealthy(str(exc))

    async def _check_graphiti(self) -> ComponentCheck:
        """Probe the graph-memory layer with a bounded, read-only query.

        Absence resolves to the null probe before this runs. Failures report
        the exception *type*, never its message: the underlying driver
        interpolates its connection URI into error text, so ``str(exc)``
        would put a DSN into the response body and the log line.
        """
        client = self.graph_memory_client
        client = _require(client, "graphiti")
        start = time.perf_counter()
        try:
            async with asyncio.timeout(_GRAPH_MEMORY_PROBE_TIMEOUT_S):
                await client.driver.execute_query(_GRAPH_MEMORY_PROBE_QUERY)
        except (Neo4jError, DriverError, OSError, TimeoutError) as exc:
            exc.add_note("component=graphiti, operation=check_graphiti")
            logger.bind(error_type=type(exc).__name__, component="graphiti").exception(
                "Graphiti health check failed"
            )
            return ComponentCheck.unhealthy(type(exc).__name__)
        response_time = (time.perf_counter() - start) * 1000
        return ComponentCheck.healthy(response_time_ms=round(response_time, 2))

    async def _check_agent_memory(self) -> AgentMemoryCheck:
        """Probe agent memory (cognee), mirroring the middleware probe's three states.

        The graph-procedure precondition (APOC/GDS) is reported as a **named sub-field**
        and does not fail the whole check: it is the only way a silently failing
        consolidation is ever observed, and its absence means consolidation refuses to
        run — not that the subsystem is down.
        """
        if self.cognee_config is None:
            return AgentMemoryCheck(status=HealthStatus.DEGRADED, state="not_configured")

        graph_procedures_available = False
        graph_reachable = False
        if self.neo4j_driver is not None:
            try:
                async with asyncio.timeout(_GRAPH_MEMORY_PROBE_TIMEOUT_S):
                    records, _, _ = await self.neo4j_driver.execute_query(
                        "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.' "
                        "OR name STARTS WITH 'gds.' RETURN count(name) AS n"
                    )
                graph_reachable = True
                # neo4j Record supports mapping access; a test double may not.
                count = (
                    records[0].get("n", 0)
                    if hasattr(records[0], "get")
                    else getattr(records[0], "n", 0)
                )
                graph_procedures_available = bool(records and count > 0)
            except (Neo4jError, DriverError, OSError, TimeoutError) as exc:
                exc.add_note("component=agent_memory, operation=check_agent_memory")
                logger.bind(error_type=type(exc).__name__, component="agent_memory").exception(
                    "Agent memory health check failed"
                )
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
            # Absent procedures mean consolidation will refuse to run; they do NOT
            # mean this check should fail.
            graph_procedures_available=graph_procedures_available,
            embedding_dimension=self.cognee_config.embedding_dimension,
        )

    async def _guarded_agent_memory(self) -> AgentMemoryCheck:
        """Run the agent-memory probe, preserving its typed shape.

        The shared runner answers timeouts/crashes with a plain
        ``ComponentCheck``; this guard re-roots that fallback into an
        ``AgentMemoryCheck`` so the DTO below always receives the subtype
        its field declares.
        """
        result = await self._run_async_check(self._check_agent_memory, "agent_memory")
        if isinstance(result, AgentMemoryCheck):
            return result
        return AgentMemoryCheck.unhealthy(
            result.error or "agent_memory probe failed", state="error"
        )

    def _check_celery(self) -> ComponentCheck:
        app = self.celery_app
        app = _require(app, "celery")
        try:
            start = time.perf_counter()
            conn = app.connection()
            try:
                conn.ensure_connection(max_retries=1, timeout=2)
            finally:
                conn.release()
            response_time = (time.perf_counter() - start) * 1000
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            exc.add_note("component=celery, operation=check_celery")
            logger.bind(error=str(exc), component="celery").exception("Celery health check failed")
            return ComponentCheck.unhealthy(str(exc))
        else:
            return ComponentCheck.healthy(response_time_ms=round(response_time, 2))

    @staticmethod
    def _check_memory() -> ComponentCheck:
        memory = psutil.virtual_memory()
        process_memory = psutil.Process().memory_info()
        details = {
            "system": {
                "total_mb": round(memory.total / 1024 / 1024, 2),
                "available_mb": round(memory.available / 1024 / 1024, 2),
                "used_mb": round(memory.used / 1024 / 1024, 2),
                "percent": round(memory.percent, 1),
            },
            "process": {
                "rss_mb": round(process_memory.rss / 1024 / 1024, 2),
                "vms_mb": round(process_memory.vms / 1024 / 1024, 2),
            },
        }
        if memory.percent >= 90:
            return ComponentCheck.warning(state="high_usage", **details)
        return ComponentCheck.healthy(state="sampled", **details)

    @staticmethod
    def _check_disk() -> ComponentCheck:
        try:
            disk = psutil.disk_usage(".")
            details = {
                "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                "percent": round(disk.percent, 1),
            }
        except (FileNotFoundError, PermissionError, OSError) as exc:
            exc.add_note("component=disk, operation=check_disk")
            logger.bind(error=str(exc), component="disk").exception("Disk health check failed")
            return ComponentCheck.unhealthy(str(exc), state="inaccessible")
        if disk.percent >= 90:
            return ComponentCheck.warning(state="high_usage", **details)
        return ComponentCheck.healthy(state="accessible", **details)

    @staticmethod
    def _severity(status: HealthStatus) -> int:
        """Rank a component status for overall aggregation (Pattern 2)."""
        match status:
            case HealthStatus.UNHEALTHY:
                return 2
            case HealthStatus.WARNING:
                return 1
            case _:
                return 0

    @classmethod
    def _compute_overall_status(cls, checks: HealthChecksDTO) -> HealthStatus:
        severities = [
            cls._severity(check.status)
            for check in (
                checks.database,
                checks.redis,
                checks.postgres,
                checks.neo4j,
                checks.graphiti,
                checks.celery,
                checks.memory,
                checks.disk,
            )
        ]
        worst = max(severities, default=0)
        if worst >= 2:
            return HealthStatus.UNHEALTHY
        if worst >= 1:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _get_application_health(self) -> dict[str, Any]:
        process = psutil.Process()
        memory_info = process.memory_info()
        uptime = time.time() - self.start_time
        return {
            "environment": get_settings().ENVIRONMENT,
            "uptime": f"{uptime:.2f} seconds",
            "memoryUsage": {
                "rss": f"{memory_info.rss / 1024 / 1024:.2f} MB",
                "vms": f"{memory_info.vms / 1024 / 1024:.2f} MB",
            },
            "pid": os.getpid(),
        }

    @staticmethod
    def _get_system_health() -> dict[str, Any]:
        # ``interval=1`` blocks the caller for a full second. The probe is
        # already dispatched to a worker thread; a non-blocking sample also
        # prevents a health request from consuming an event-loop second.
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        try:
            get_load_average = cast(
                "Callable[[], tuple[float, float, float]]",
                psutil.__dict__["getloadavg"],
            )
            load_avg = list(get_load_average())
        except (AttributeError, OSError) as exc:
            exc.add_note("component=system, operation=system_health")
            logger.bind(operation="system_health").warning("System load average unavailable")
            load_avg = [0.0, 0.0, 0.0]
        return {
            "cpuUsage": load_avg,
            "cpuUsagePercent": f"{cpu_percent:.2f}%",
            "totalMemory": f"{memory.total / 1024 / 1024:.2f} MB",
            "freeMemory": f"{memory.available / 1024 / 1024:.2f} MB",
            "platform": platform.system(),
            "arch": platform.machine(),
        }
