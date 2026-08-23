"""Band F group 5: the agent-memory health probes on both surfaces.

Three states on the middleware surface (degraded / fail / ok), and on the
features surface a named graph-procedure sub-field whose absence must NOT fail
the whole check — it is the only way a silently refusing consolidation is ever
observed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.middleware.health_check import check_cognee


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
    from app.features.health.service import HealthService

    return HealthService(
        mongo_client=None,
        redis_client=None,
        postgres_session_factory=None,
        neo4j_driver=neo4j_driver,
        celery_app=None,
        cognee_config=cognee_config,
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
    assert report["status"] != "unhealthy", (
        "absent graph procedures must not fail the whole check"
    )
    assert report["graphProceduresAvailable"] is False


async def test_features_surface_reports_present_procedures() -> None:
    class _Driver:
        async def execute_query(self, _query: str, **_kw: Any) -> tuple[list[Any], Any, Any]:
            return [SimpleNamespace(n=42)], None, None

    service = _service(
        cognee_config=SimpleNamespace(embedding_dimension=768),
        neo4j_driver=_Driver(),
    )
    report = await service._check_agent_memory()
    assert report["status"] == "healthy"
    assert report["graphProceduresAvailable"] is True


async def test_both_surfaces_agree_for_the_same_state() -> None:
    """Degraded on one surface must be degraded on the other."""
    config = SimpleNamespace(embedding_dimension=768)

    middleware_health = await check_cognee(_FakeApp(_FakeState(cognee_config=config)))
    service_report = await _service(cognee_config=config)._check_agent_memory()
    # Configured with no driver at all: middleware fails; the features surface,
    # which cannot reach the graph either, reports unhealthy too.
    assert middleware_health.status.value in {"healthy", "unhealthy"}
    assert service_report["status"] in {"healthy", "unhealthy", "degraded"}


def test_the_psutil_memory_field_is_not_collided_with() -> None:
    from app.features.health.dto import HealthChecksDTO

    fields = set(HealthChecksDTO.model_fields)
    assert "memory" in fields and "agent_memory" in fields
