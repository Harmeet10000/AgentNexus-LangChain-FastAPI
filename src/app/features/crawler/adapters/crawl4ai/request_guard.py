"""Browser-level destination rules for the crawler egress boundary.

The proxy's connection-time address policy is the security boundary. These
rules are defense in depth: they deny destinations the crawler must never
request — a scheme or port outside HTTP/80 and HTTPS/443 — before the
request is handed to the proxy at all, and they cover browser subresources
that never pass through application-level URL validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from app.features.crawler.application.egress import NON_NETWORK_SCHEMES, CrawlerEgressConfigError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from crawl4ai import AsyncWebCrawler
    from playwright.async_api import BrowserContext, Page, Route

    from app.features.crawler.application.egress import CrawlerEgressPolicy


def evaluate_browser_target(url: str, policy: CrawlerEgressPolicy) -> bool:
    """Return True when the crawler may request this URL."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError:
        return False
    if scheme in NON_NETWORK_SCHEMES:
        return True
    return policy.is_browser_target_allowed(scheme, port)


def build_route_guard(
    policy: CrawlerEgressPolicy,
) -> Callable[[Route], Awaitable[None]]:
    """Build the Playwright route handler that denies unsupported destinations."""

    async def _guard(route: Route) -> None:
        if evaluate_browser_target(route.request.url, policy):
            await route.continue_()
            return
        await route.abort("blockedbyclient")

    return _guard


def install_egress_request_guard(
    crawler: AsyncWebCrawler,
    policy: CrawlerEgressPolicy,
) -> None:
    """Deny unsupported destinations on every page context the browser creates.

    The interception is installed as a browser hook rather than per request so
    that navigation *and* subresources are covered by the same rule.
    """
    guard = build_route_guard(policy)

    async def _on_page_context_created(
        _page: Page,
        *,
        context: BrowserContext | None = None,
        **_kwargs: object,
    ) -> None:
        if context is None:
            return
        await context.route("**/*", guard)

    strategy = getattr(crawler, "crawler_strategy", None)
    set_hook = getattr(strategy, "set_hook", None)
    if set_hook is None:
        message = "the crawl4ai strategy does not support browser request interception"
        raise CrawlerEgressConfigError(message)
    set_hook("on_page_context_created", _on_page_context_created)
