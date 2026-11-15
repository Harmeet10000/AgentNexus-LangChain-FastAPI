"""Task 4.3 — crawler egress failures fail closed at every boundary.

These tests prove unsupported browser targets are rejected before proxy I/O,
proxy outages never trigger direct navigation, loopback must be explicitly
denied, and an unavailable preflight prevents browser creation.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from app.features.crawler.application.egress import CrawlerEgressPolicy
from app.utils import ServiceUnavailableException
from app.worker.capabilities import crawler as crawler_capability


def _policy() -> CrawlerEgressPolicy:
    return CrawlerEgressPolicy(
        proxy_url="http://crawler-egress-proxy:4750",
        public_probe_url="https://example.com/",
        loopback_probe_url="http://127.0.0.1:9/",
        preflight_timeout_seconds=5.0,
        allowed_scheme_ports={"http": (80,), "https": (443,)},
    )


@pytest.mark.asyncio
async def test_unsupported_browser_targets_are_refused_before_proxy_io() -> None:
    from app.features.crawler.adapters.crawl4ai.request_guard import (
        build_route_guard,
        evaluate_browser_target,
    )

    seen: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)):
        for url in ("ftp://example.com/", "http://example.com:8080/"):
            assert evaluate_browser_target(url, _policy()) is False
            route = _Route(url)
            await build_route_guard(_policy())(route)
            assert route.aborted is True

    assert seen == []


@pytest.mark.asyncio
async def test_proxy_outage_does_not_attempt_direct_request() -> None:
    from app.features.crawler.adapters.proxy.preflight import run_egress_preflight

    seen: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        message = "proxy unavailable"
        raise httpx.ConnectError(message, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await run_egress_preflight(_policy(), client=client)

    assert result.available is False
    assert seen
    assert all(request.url.host == "example.com" for request in seen)


@pytest.mark.asyncio
async def test_permitted_loopback_probe_fails_closed() -> None:
    from app.features.crawler.adapters.proxy.preflight import run_egress_preflight

    def transport(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await run_egress_preflight(_policy(), client=client)

    assert result.available is False
    assert result.denial_status == 200


@pytest.mark.asyncio
async def test_unavailable_preflight_prevents_browser_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.crawler.adapters.proxy.preflight import EgressPreflightResult

    async def unavailable(_policy: CrawlerEgressPolicy) -> EgressPreflightResult:
        return EgressPreflightResult(available=False, reason="boundary unavailable")

    async def browser_factory(**_kwargs: str) -> object:
        message = "browser must not be created"
        raise AssertionError(message)

    monkeypatch.setattr(crawler_capability, "_resolve_egress_policy", _policy)
    monkeypatch.setattr(crawler_capability, "run_egress_preflight", unavailable)
    monkeypatch.setattr(crawler_capability, "create_crawl4ai_crawler", browser_factory)

    with pytest.raises(ServiceUnavailableException):
        await crawler_capability._provision_crawler_worker()


class _Route:
    def __init__(self, url: str) -> None:
        self.request = SimpleNamespace(url=url)
        self.aborted = False

    async def abort(self, _error_code: str | None = None) -> None:
        self.aborted = True

    async def continue_(self) -> None:
        message = "unsupported target must not continue"
        raise AssertionError(message)
