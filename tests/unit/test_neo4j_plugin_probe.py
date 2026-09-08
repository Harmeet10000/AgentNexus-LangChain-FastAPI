"""Neo4j plugin probe: APOC/GDS availability is reported, never silent."""

from __future__ import annotations

from typing import Any

from app.features.health.health_check import check_neo4j_plugins


class _FakeState:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeApp:
    def __init__(self, state: _FakeState) -> None:
        self.state = state


class _FakeResult:
    def __init__(self, n: int) -> None:
        self._n = n

    async def single(self) -> dict[str, int]:
        return {"n": self._n}


class _FakeSession:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def run(self, query: str, params: dict[str, str]) -> _FakeResult:
        return _FakeResult(self._counts[params["prefix"]])


class _FakeDriver:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def session(self) -> _FakeSession:
        return _FakeSession(self._counts)


async def test_uninitialised_driver_is_degraded() -> None:
    health = await check_neo4j_plugins(_FakeApp(_FakeState(neo4j_driver=None)))
    assert health.name == "neo4j-plugins"
    assert health.status.value == "degraded"


async def test_bare_neo4j_without_apoc_fails() -> None:
    app = _FakeApp(_FakeState(neo4j_driver=_FakeDriver({"apoc.": 0, "gds.": 0})))
    health = await check_neo4j_plugins(app)
    assert health.status.value == "unhealthy"
    assert health.message is not None and "APOC" in health.message


async def test_missing_gds_only_degrades() -> None:
    app = _FakeApp(_FakeState(neo4j_driver=_FakeDriver({"apoc.": 162, "gds.": 0})))
    health = await check_neo4j_plugins(app)
    assert health.status.value == "degraded"
    assert health.message is not None and "GDS" in health.message


async def test_fully_provisioned_neo4j_is_ok() -> None:
    app = _FakeApp(_FakeState(neo4j_driver=_FakeDriver({"apoc.": 162, "gds.": 446})))
    health = await check_neo4j_plugins(app)
    assert health.status.value == "healthy"
