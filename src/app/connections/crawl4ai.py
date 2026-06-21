"""Crawl4AI browser initialization and dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crawl4ai import AsyncWebCrawler, BrowserConfig
from fastapi.requests import HTTPConnection

from app.config import get_settings
from app.shared.crawler import WebCrawler

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from app.config.settings import Settings


async def create_crawl4ai_crawler() -> AsyncWebCrawler:
    """Create and start a Crawl4AI browser for lifespan management.

    Raises:
        Exception: Propagates any browser-launch error to lifespan caller.
    """
    settings: Settings = get_settings()
    crawler = AsyncWebCrawler(
        config=BrowserConfig(headless=settings.CRAWL4AI_HEADLESS),
    )
    await crawler.start()
    return crawler


async def close_crawl4ai_crawler(crawler: AsyncWebCrawler | None) -> None:
    """Close the Crawl4AI browser during lifespan shutdown."""
    if crawler is not None:
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
    return WebCrawler(redis_client=redis_client)
