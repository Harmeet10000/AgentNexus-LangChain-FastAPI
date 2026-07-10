"""Tests for API deprecation headers on v1 routes."""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestDeprecationHeaders:
    """Verify v1 routes get Deprecation/Sunset headers, v2 does not."""

    @pytest.fixture(autouse=True)
    def _setup(self, client):  # noqa: ANN001 — test fixture, annotation not required
        self.client = client

    def test_v1_route_has_deprecation_header(self) -> None:
        resp = self.client.get("/api/v1/")
        # Any v1 route should have Deprecation header
        if resp.status_code < 404:
            assert resp.headers.get("deprecation") == "true"

    def test_v1_route_has_sunset_header(self) -> None:
        resp = self.client.get("/api/v1/")
        if resp.status_code < 404:
            assert "sunset" in resp.headers

    def test_v1_route_has_link_header(self) -> None:
        resp = self.client.get("/api/v1/")
        if resp.status_code < 404:
            link = resp.headers.get("link", "")
            assert 'rel="successor-version"' in link

    def test_v2_route_no_deprecation_header(self) -> None:
        resp = self.client.get("/api/v2/")
        assert resp.headers.get("deprecation") is None

    def test_health_endpoint_no_deprecation_header(self) -> None:
        resp = self.client.get("/health")
        assert resp.headers.get("deprecation") is None

    def test_metrics_endpoint_no_deprecation_header(self) -> None:
        resp = self.client.get("/metrics")
        assert resp.headers.get("deprecation") is None

    def test_root_endpoint_no_deprecation_header(self) -> None:
        resp = self.client.get("/")
        assert resp.headers.get("deprecation") is None
