"""Task 3.2 — the browser may only request HTTP/80 and HTTPS/443.

The proxy's connection-time address policy is the security boundary; the
browser rule is defense in depth that denies unsupported destinations before
they are handed to the proxy at all, and it covers subresources that never
pass through application-level URL validation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from app.features.crawler.application.egress import CrawlerEgressPolicy

if TYPE_CHECKING:
    from typing import Any

pytestmark = pytest.mark.unit


def _policy(**overrides: Any) -> CrawlerEgressPolicy:
    values: dict[str, Any] = {
        "proxy_url": "http://crawler-egress-proxy:4750",
        "public_probe_url": "https://example.com/",
        "loopback_probe_url": "http://127.0.0.1:9/",
        "preflight_timeout_seconds": 5.0,
        "allowed_scheme_ports": {"http": (80,), "https": (443,)},
        "bypass_destinations": ("rabbitmq",),
    }
    values.update(overrides)
    return CrawlerEgressPolicy(**values)


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("http://example.com/", True),
        ("http://example.com:80/index.html", True),
        ("https://example.com/", True),
        ("https://example.com:443/path", True),
        ("http://example.com:8080/", False),
        ("https://example.com:8443/", False),
        ("ftp://example.com/", False),
        ("file:///etc/passwd", False),
        ("data:text/html,<b>x</b>", False),
        ("ws://example.com/socket", False),
    ],
)
def test_only_supported_destinations_are_allowed(url: str, allowed: bool) -> None:
    from app.features.crawler.adapters.crawl4ai.request_guard import evaluate_browser_target

    assert evaluate_browser_target(url, _policy()) is allowed


def test_browser_probe_target_is_not_denied() -> None:
    from app.features.crawler.adapters.crawl4ai.request_guard import evaluate_browser_target

    assert evaluate_browser_target("about:blank", _policy()) is True
    assert evaluate_browser_target("about:srcdoc", _policy()) is False


def test_malformed_url_is_denied() -> None:
    from app.features.crawler.adapters.crawl4ai.request_guard import evaluate_browser_target

    assert evaluate_browser_target("http://example.com:not-a-port/", _policy()) is False


class _FakeRoute:
    """Records what the guard did with one request."""

    def __init__(self, url: str) -> None:
        self.request = SimpleNamespace(url=url)
        self.aborted = False
        self.continued = False

    async def abort(self, error_code: str | None = None) -> None:
        self.aborted = True

    async def continue_(self) -> None:
        self.continued = True


class _FakeContext:
    def __init__(self) -> None:
        self.patterns: list[str] = []
        self.handler: Any = None

    async def route(self, pattern: str, handler: Any) -> None:
        self.patterns.append(pattern)
        self.handler = handler


class _FakeStrategy:
    def __init__(self) -> None:
        self.hooks: dict[str, Any] = {}

    def set_hook(self, hook_type: str, hook: Any) -> None:
        self.hooks[hook_type] = hook


class _FakeCrawler:
    def __init__(self) -> None:
        self.crawler_strategy = _FakeStrategy()


@pytest.mark.asyncio
async def test_route_guard_aborts_unsupported_destinations() -> None:
    from app.features.crawler.adapters.crawl4ai.request_guard import build_route_guard

    guard = build_route_guard(_policy())

    denied = _FakeRoute("http://example.com:8080/")
    await guard(denied)
    assert denied.aborted is True
    assert denied.continued is False

    allowed = _FakeRoute("https://example.com/")
    await guard(allowed)
    assert allowed.continued is True
    assert allowed.aborted is False


@pytest.mark.asyncio
async def test_interception_is_installed_on_every_page_context() -> None:
    from app.features.crawler.adapters.crawl4ai.request_guard import install_egress_request_guard

    crawler = _FakeCrawler()
    install_egress_request_guard(crawler, _policy())

    hook = crawler.crawler_strategy.hooks["on_page_context_created"]
    context = _FakeContext()
    await hook(object(), context=context, config=None)

    assert context.patterns == ["**/*"]
    assert context.handler is not None


def test_missing_interception_support_fails_closed() -> None:
    from app.features.crawler.adapters.crawl4ai.request_guard import install_egress_request_guard
    from app.features.crawler.application.egress import CrawlerEgressConfigError

    class _NoHookStrategy:
        pass

    crawler = SimpleNamespace(crawler_strategy=_NoHookStrategy())

    with pytest.raises(CrawlerEgressConfigError):
        install_egress_request_guard(crawler, _policy())
