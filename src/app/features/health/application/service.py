"""Application orchestration for health checks."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from app.features.health.application.dto.result import HealthStatus

from .dto.aggregate import HealthChecksDTO, HealthDataDTO, HealthResultDTO
from .dto.result import AgentMemoryCheck, ComponentCheck

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from .ports.probes import Probe

_HEALTH_CHECK_TIMEOUT_S = 8.0


class _NullProbe:
    """Null object for an optional resource absent from application state."""

    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    async def check() -> ComponentCheck:
        return ComponentCheck.not_configured()


class HealthService:
    """Run named probes and aggregate their typed results.

    Resource construction is intentionally absent from this class. A process
    composition root creates adapter probes and injects them here; this class
    owns only scheduling, timeout policy, and health semantics.
    """

    def __init__(
        self,
        probes: Iterable[Probe],
        *,
        environment: str = "unknown",
        start_time: float | None = None,
    ) -> None:
        self._probes = {probe.name: probe for probe in probes}
        self.environment = environment
        self.start_time = time.time() if start_time is None else start_time

    @staticmethod
    async def get_self_info(
        server_name: str,
        server_version: str,
        client_host: str,
    ) -> dict[str, Any]:
        """Return transport-neutral service metadata."""
        return {
            "server": server_name,
            "version": server_version,
            "client": client_host,
            "timestamp": time.time(),
        }

    def _get_probe(self, name: str) -> Probe:
        return self._probes.get(name, _NullProbe(name))

    async def _run_probe(self, name: str) -> ComponentCheck:
        probe = self._get_probe(name)
        try:
            async with asyncio.timeout(_HEALTH_CHECK_TIMEOUT_S):
                result = await probe.check()
        except TimeoutError as exc:
            exc.add_note(f"component={name}, operation=health_check")
            return ComponentCheck.timeout(name, _HEALTH_CHECK_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — health must fail closed
            exc.add_note(f"component={name}, operation=health_check")
            return ComponentCheck.unhealthy(type(exc).__name__, state="error")
        return result

    async def _check_agent_memory(self) -> AgentMemoryCheck:
        """Preserve the named agent-memory result type at the application boundary."""
        result = await self._run_probe("agent_memory")
        if isinstance(result, AgentMemoryCheck):
            return result
        return AgentMemoryCheck.unhealthy(
            result.error or "agent_memory probe failed", state="error"
        )

    async def get_health(self) -> HealthResultDTO:
        """Run independent probes concurrently and return the aggregate."""
        names = (
            "mongodb",
            "redis",
            "postgres",
            "neo4j",
            "graphiti",
            "agent_memory",
            "celery",
            "memory",
            "disk",
            "runtime",
        )
        results = dict(
            zip(
                names, await asyncio.gather(*(self._run_probe(name) for name in names)), strict=True
            )
        )

        checks = HealthChecksDTO(
            database=results["mongodb"],
            redis=results["redis"],
            postgres=results["postgres"],
            neo4j=results["neo4j"],
            graphiti=results["graphiti"],
            celery=results["celery"],
            memory=results["memory"],
            disk=results["disk"],
            agent_memory=(
                results["agent_memory"]
                if isinstance(results["agent_memory"], AgentMemoryCheck)
                else AgentMemoryCheck.unhealthy(
                    results["agent_memory"].error or "agent_memory probe failed", state="error"
                )
            ),
        )
        overall_status = self._compute_overall_status(checks)
        status_code = 200 if overall_status == HealthStatus.HEALTHY else 503
        runtime_details = results["runtime"].details
        application_health = runtime_details.get("application", {})
        system_health = runtime_details.get("system", {})
        data = HealthDataDTO(
            status=overall_status,
            timestamp=time.time(),
            application=application_health,
            system=system_health,
            checks=checks,
        )
        return HealthResultDTO(
            message=f"Health check: {overall_status}",
            status_code=status_code,
            data=data,
        )

    @staticmethod
    def _severity(status: HealthStatus) -> int:
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
