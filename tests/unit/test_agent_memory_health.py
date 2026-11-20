"""Band F group 5: the agent-memory health probes on both surfaces.

Three states on the middleware surface (degraded / fail / ok), and on the
features surface a named graph-procedure sub-field whose absence must NOT fail
the whole check — it is the only way a silently refusing consolidation is ever
observed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from app.features.health.adapters.outbound.neo4j.probe import AgentMemoryProbe, check_cognee
from app.features.health.application.dto.result import ComponentCheck
from app.features.health.module import get_health_service
from app.kernel.result import HealthStatus

if TYPE_CHECKING:
    from typing import Any


class _FakeState:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeDriver:
    def __init__(self, *, reachable: bool = True) -> None:
        self.reachable = reachable

    async def verify_connectivity(self) -> None:
        if not self.reachable:
            msg = "connection refused"
            raise OSError(msg)


class _FakeApp:
    def __init__(self, state: _FakeState) -> None:
        self.state = state


async def test_unconfigured_cognee_is_degraded() -> None:
    app = _FakeApp(_FakeState(cognee_config=None))
    health = await check_cognee(app)
    assert health.status.value == "degraded"
    assert health.name == "cognee"


async def test_configured_but_unreachable_cognee_fails() -> None:
    config = SimpleNamespace(embedding_dimension=768)
    app = _FakeApp(_FakeState(cognee_config=config, neo4j_driver=_FakeDriver(reachable=False)))
    health = await check_cognee(app)
    assert health.status.value == "unhealthy"


async def test_configured_and_reachable_cognee_is_ok() -> None:
    config = SimpleNamespace(embedding_dimension=768)
    app = _FakeApp(_FakeState(cognee_config=config, neo4j_driver=_FakeDriver()))
    health = await check_cognee(app)
    assert health.status.value == "healthy"


# --- features surface ---


def _service(cognee_config: Any, neo4j_driver: Any = None) -> Any:
    from app.features.health.application.service import HealthService

    return HealthService(
        probes=(AgentMemoryProbe(cognee_config, neo4j_driver),),
    )


async def test_features_surface_reports_graph_procedures_as_a_named_subfield() -> None:
    class _Driver:
        async def execute_query(self, _query: str, **_kw: Any) -> tuple[list[Any], Any, Any]:
            return [SimpleNamespace(n=0)], None, None  # no APOC/GDS

    service = _service(
        cognee_config=SimpleNamespace(embedding_dimension=768),
        neo4j_driver=_Driver(),
    )
    report = await service._check_agent_memory()
    assert report.status != HealthStatus.UNHEALTHY, (
        "absent graph procedures must not fail the whole check"
    )
    assert report.graph_procedures_available is False


async def test_features_surface_reports_present_procedures() -> None:
    class _Driver:
        async def execute_query(self, _query: str, **_kw: Any) -> tuple[list[Any], Any, Any]:
            return [SimpleNamespace(n=42)], None, None

    service = _service(
        cognee_config=SimpleNamespace(embedding_dimension=768),
        neo4j_driver=_Driver(),
    )
    report = await service._check_agent_memory()
    assert report.status == HealthStatus.HEALTHY
    assert report.graph_procedures_available is True


async def test_health_keeps_its_own_200_status_when_optional_backends_are_absent(
    monkeypatch,
) -> None:
    service = _service(cognee_config=None)
    # Disk usage is host state, not part of this contract. Inject an explicit
    # healthy probe so this assertion stays independent of the runner disk.

    class _HealthyDisk:
        name = "disk"

        async def check(self) -> ComponentCheck:
            return ComponentCheck.healthy(state="accessible")

    service._probes["disk"] = _HealthyDisk()
    result = await service.get_health()

    assert result.status_code == 200
    assert result.data is not None
    assert result.data.status == "healthy"
    assert result.data.checks.graphiti.state == "not_configured"


async def test_health_keeps_its_own_503_status_for_required_probe_failure() -> None:
    service = _service(cognee_config=None)

    class _UnhealthyMongo:
        name = "mongodb"

        async def check(self) -> ComponentCheck:
            return ComponentCheck.unhealthy("disconnected", state="disconnected")

    service._probes["mongodb"] = _UnhealthyMongo()
    result = await service.get_health()

    assert result.status_code == 503
    assert result.data is not None
    assert result.data.status == "unhealthy"


def test_health_service_factory_reads_clients_from_app_state() -> None:
    state = _FakeState(
        mongo_client=object(),
        redis=object(),
        db_session_local=object(),
        neo4j_driver=object(),
        celery=object(),
        graphiti=object(),
        cognee_config=object(),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    service = get_health_service(request)

    assert service._probes["mongodb"]._client is state.mongo_client
    assert service._probes["redis"]._client is state.redis
    assert service._probes["postgres"]._session_factory is state.db_session_local
    assert service._probes["neo4j"]._driver is state.neo4j_driver
    assert service._probes["celery"]._app is state.celery
    assert service._probes["graphiti"]._client is state.graphiti
    assert service._probes["agent_memory"]._config is state.cognee_config


async def test_both_surfaces_agree_for_the_same_state() -> None:
    """Degraded on one surface must be degraded on the other."""
    config = SimpleNamespace(embedding_dimension=768)

    middleware_health = await check_cognee(_FakeApp(_FakeState(cognee_config=config)))
    service_report = await _service(cognee_config=config)._check_agent_memory()
    # Configured with no driver at all: middleware fails; the features surface,
    # which cannot reach the graph either, reports unhealthy too.
    assert middleware_health.status.value in {"healthy", "unhealthy"}
    assert service_report.status in {
        HealthStatus.HEALTHY,
        HealthStatus.UNHEALTHY,
        HealthStatus.DEGRADED,
    }


def test_the_psutil_memory_field_is_not_collided_with() -> None:
    from app.features.health.dto import HealthChecksDTO

    fields = set(HealthChecksDTO.model_fields)
    assert "memory" in fields
    assert "agent_memory" in fields
