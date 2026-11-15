"""Task 3.3 — the capability preflight proves the boundary before provisioning.

The crawler worker may only run when the proxy both permits an operator
controlled public probe and explicitly denies a safe loopback probe. A
TCP-open proxy is not enough: a misconfigured policy still exposes private
infrastructure, and an unreachable proxy must never degrade the worker to
direct networking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from app.features.crawler.application.egress import CrawlerEgressPolicy

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit

_DENIAL_HEADERS = {"X-Smokescreen-Error": "Egress proxying is denied to host '127.0.0.1:9'"}


def _policy() -> CrawlerEgressPolicy:
    return CrawlerEgressPolicy(
        proxy_url="http://crawler-egress-proxy:4750",
        public_probe_url="https://example.com/",
        loopback_probe_url="http://127.0.0.1:9/",
        preflight_timeout_seconds=5.0,
        allowed_scheme_ports={"http": (80,), "https": (443,)},
    )


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_public_probe_allowed_and_loopback_denied_provisions_capability() -> None:
    from app.features.crawler.adapters.proxy.preflight import run_egress_preflight

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            return httpx.Response(407, headers=_DENIAL_HEADERS, text="denied")
        return httpx.Response(200, text="ok")

    async with _client(handler) as client:
        result = await run_egress_preflight(_policy(), client=client)

    assert result.available is True
    assert result.public_probe_status == 200
    assert result.denial_status == 407


@pytest.mark.asyncio
async def test_loopback_probe_that_is_permitted_fails_closed() -> None:
    from app.features.crawler.adapters.proxy.preflight import run_egress_preflight

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    async with _client(handler) as client:
        result = await run_egress_preflight(_policy(), client=client)

    assert result.available is False
    assert "not explicitly denied" in result.reason


@pytest.mark.asyncio
async def test_ambiguous_denial_without_the_proxy_error_header_fails_closed() -> None:
    from app.features.crawler.adapters.proxy.preflight import run_egress_preflight

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            return httpx.Response(502, text="Bad gateway")
        return httpx.Response(200, text="ok")

    async with _client(handler) as client:
        result = await run_egress_preflight(_policy(), client=client)

    assert result.available is False


@pytest.mark.asyncio
async def test_failing_public_probe_fails_closed() -> None:
    from app.features.crawler.adapters.proxy.preflight import run_egress_preflight

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    async with _client(handler) as client:
        result = await run_egress_preflight(_policy(), client=client)

    assert result.available is False
    assert result.public_probe_status == 503


@pytest.mark.asyncio
async def test_unreachable_proxy_fails_closed_without_a_direct_fallback() -> None:
    from app.features.crawler.adapters.proxy.preflight import run_egress_preflight

    def handler(request: httpx.Request) -> httpx.Response:
        message = "proxy unreachable"
        raise httpx.ConnectError(message, request=request)

    async with _client(handler) as client:
        result = await run_egress_preflight(_policy(), client=client)

    assert result.available is False
    assert "could not reach the egress boundary" in result.reason
