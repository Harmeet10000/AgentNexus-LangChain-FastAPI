"""Contract and aggregation tests for the health application boundary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from app.features.health.application.dto.result import (
    AgentMemoryCheck,
    ComponentCheck,
    HealthStatus,
)
from app.features.health.application.service import HealthService

if TYPE_CHECKING:
    from app.features.health.application.ports.probes import Probe

pytestmark = pytest.mark.unit


class _Probe:
    def __init__(self, name: str, result: ComponentCheck, delay: float = 0.0) -> None:
        self.name = name
        self._result = result
        self._delay = delay

    async def check(self) -> ComponentCheck:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._result


def test_probe_contract_is_named_and_async() -> None:
    probe: Probe = _Probe("redis", ComponentCheck.healthy())

    assert probe.name == "redis"
    result = asyncio.run(probe.check())
    assert isinstance(result, ComponentCheck)
    assert result.status is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_service_runs_probes_concurrently_and_aggregates_by_name() -> None:
    service = HealthService(
        probes=(
            _Probe("mongodb", ComponentCheck.unhealthy("down"), delay=0.01),
            _Probe("agent_memory", AgentMemoryCheck(status="healthy", state="configured")),
            _Probe("runtime", ComponentCheck.healthy(application={"pid": 7}, system={})),
        )
    )

    result = await service.get_health()

    assert result.status_code == 503
    assert result.data.status is HealthStatus.UNHEALTHY
    assert result.data.checks.database.error == "down"
    assert result.data.checks.agent_memory.graph_reachable is False
    assert result.data.application == {"pid": 7}
    assert result.data.checks.redis.status is HealthStatus.UNKNOWN


@pytest.mark.asyncio
async def test_optional_unknown_probes_do_not_degrade_aggregate() -> None:
    service = HealthService(
        probes=(_Probe("runtime", ComponentCheck.healthy(application={}, system={})),)
    )

    result = await service.get_health()

    assert result.status_code == 200
    assert result.data.status is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_probe_exception_is_normalized_to_unhealthy_result() -> None:
    class _BrokenProbe:
        name = "redis"

        async def check(self) -> ComponentCheck:
            message = "driver failure"
            raise RuntimeError(message)

    result = await HealthService(probes=(_BrokenProbe(),)).get_health()

    assert result.status_code == 503
    assert result.data.checks.redis.status is HealthStatus.UNHEALTHY
    assert result.data.checks.redis.error == "RuntimeError"
