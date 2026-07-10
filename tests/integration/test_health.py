"""Integration tests for GET /health endpoint.

These tests require a running stack (PostgreSQL, Redis, MongoDB).
Run with: uv run pytest tests/integration/test_health.py -v
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestHealthEndpoint:
    """Verify /health returns correct status codes and dependency info."""

    @pytest.fixture(autouse=True)
    def _setup(self, client):  # noqa: ANN001 — test fixture, annotation not required
        self.client = client

    def test_healthy_returns_200(self) -> None:
        resp = self.client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")

    def test_healthy_has_version(self) -> None:
        resp = self.client.get("/health")
        body = resp.json()
        assert "version" in body

    def test_healthy_has_dependencies_list(self) -> None:
        resp = self.client.get("/health")
        body = resp.json()
        assert isinstance(body["dependencies"], list)
        assert len(body["dependencies"]) >= 1

    def test_dependency_has_required_fields(self) -> None:
        resp = self.client.get("/health")
        body = resp.json()
        for dep in body["dependencies"]:
            assert "name" in dep
            assert "status" in dep
            assert dep["status"] in ("healthy", "degraded", "unhealthy")
            assert "latency_ms" in dep

    def test_postgres_dependency_present(self) -> None:
        resp = self.client.get("/health")
        body = resp.json()
        names = [d["name"] for d in body["dependencies"]]
        assert "postgres" in names

    def test_redis_dependency_present(self) -> None:
        resp = self.client.get("/health")
        body = resp.json()
        names = [d["name"] for d in body["dependencies"]]
        assert "redis" in names
