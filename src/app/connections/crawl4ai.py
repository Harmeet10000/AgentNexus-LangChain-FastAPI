"""Crawl4AI browser initialization and dependency injection."""

from __future__ import annotations

import os
import pathlib
import sys
from contextlib import suppress
from typing import TYPE_CHECKING

from crawl4ai import AsyncWebCrawler, BrowserConfig
from fastapi.requests import HTTPConnection

from app.shared.crawler import get_crawler_config

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from app.shared.crawler import WebCrawler
    from app.shared.crawler.config import CrawlerConfig


def _ensure_playwright_platform_override() -> None:
    """Playwright refuses to resolve browsers on Ubuntu > 24.04 (unsupported tag).

    The ubuntu-24.04 build is glibc-forward-compatible, so point the resolver at
    it when no explicit override is set. Only on Linux, only when unset — an
    operator's own override always wins.
    """
    if sys.platform != "linux" or os.environ.get("PLAYWRIGHT_HOST_PLATFORM_OVERRIDE"):
        return
    try:
        version_id = ""
        with pathlib.Path("/etc/os-release").open(encoding="ascii") as release:
            for line in release:
                if line.startswith("VERSION_ID="):
                    version_id = line.split("=", maxsplit=1)[1].strip().strip('"')
                    break
        major = int(version_id.split(".")[0])
    except (OSError, ValueError, IndexError):
        return
    if major > 24:
        os.environ["PLAYWRIGHT_HOST_PLATFORM_OVERRIDE"] = "ubuntu24.04-x64"


_ensure_playwright_platform_override()


async def create_crawl4ai_crawler() -> AsyncWebCrawler:
    """Create and start a Crawl4AI browser for lifespan management.

    Uses full CrawlerConfig for consistent BrowserConfig across all paths.

    Raises:
        Exception: Propagates any browser-launch error to lifespan caller.
    """

    config: CrawlerConfig = get_crawler_config()
    crawler = AsyncWebCrawler(
        config=BrowserConfig(**config.to_browser_config()),
    )
    await crawler.start()
    return crawler


async def close_crawl4ai_crawler(crawler: AsyncWebCrawler | None) -> None:
    """Close the Crawl4AI browser during lifespan shutdown."""
    if crawler is not None:
        with suppress(RuntimeError):
            await crawler.close()


def get_crawl4ai_crawler(connection: HTTPConnection) -> AsyncWebCrawler | None:
    """Dependency to inject Crawl4AI crawler from lifespan."""
    return getattr(connection.app.state, "crawl4ai_crawler", None)


async def get_crawler(redis_client: Redis | None = None) -> WebCrawler:
    """Get a WebCrawler domain service instance.

    Creates a new WebCrawler with optional Redis for caching.
    The underlying AsyncWebCrawler browser is created per-crawl call
    (context-managed), not from the lifespan-managed instance.
    """
    from app.shared.crawler import (
        WebCrawler,
    )

    return WebCrawler(redis_client=redis_client)
