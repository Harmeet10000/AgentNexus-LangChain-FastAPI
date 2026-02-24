"""Core crawler module using Crawl4AI."""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urldefrag

import redis.asyncio as redis
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from app.config.settings import get_settings
from app.shared.crawler.config import CrawlerConfig, get_crawler_config
from app.shared.crawler.validator import is_valid_url, sanitize_url


@dataclass
class CrawlResult:
    """Result from crawling a URL."""

    url: str
    success: bool
    markdown: str | None = None
    html: str | None = None
    title: str | None = None
    links: list[dict[str, Any]] | None = None
    error_message: str | None = None
    crawl_time_ms: int | None = None
    word_count: int | None = None
    cached: bool = False


class WebCrawler:
    """Web crawler using Crawl4AI with caching and rate limiting."""

    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or get_crawler_config()
        self._redis: redis.Redis | None = None

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            settings = get_settings()
            self._redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True,
            )
        return self._redis

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"crawl:cache:{url_hash}"

    async def _get_from_cache(self, url: str) -> CrawlResult | None:
        """Get cached crawl result."""
        try:
            settings = get_settings()
            if not settings.CRAWL4AI_PROXY:
                return None

            redis_client = await self._get_redis()
            cache_key = self._get_cache_key(url)

            cached = await redis_client.get(cache_key)
            if cached:
                import json

                data = json.loads(cached)
                result = CrawlResult(
                    url=data["url"],
                    success=data["success"],
                    markdown=data.get("markdown"),
                    html=data.get("html"),
                    title=data.get("title"),
                    links=data.get("links"),
                    error_message=data.get("error_message"),
                    crawl_time_ms=data.get("crawl_time_ms"),
                    word_count=data.get("word_count"),
                    cached=True,
                )
                return result
        except Exception:
            pass
        return None

    async def _save_to_cache(self, url: str, result: CrawlResult):
        """Save crawl result to cache."""
        try:
            settings = get_settings()
            redis_client = await self._get_redis()
            cache_key = self._get_cache_key(url)

            import json

            data = {
                "url": result.url,
                "success": result.success,
                "markdown": result.markdown,
                "html": result.html,
                "title": result.title,
                "links": result.links,
                "error_message": result.error_message,
                "crawl_time_ms": result.crawl_time_ms,
                "word_count": result.word_count,
            }

            await redis_client.setex(
                cache_key,
                settings.REDIS_CRAWL_CACHE_TTL,
                json.dumps(data),
            )
        except Exception:
            pass

    async def crawl(
        self,
        url: str,
        use_proxy: bool = False,
        bypass_cache: bool = False,
    ) -> CrawlResult:
        """
        Crawl a single URL.

        Args:
            url: URL to crawl
            use_proxy: Whether to use proxy
            bypass_cache: Whether to bypass cache

        Returns:
            CrawlResult with crawled content
        """
        url = sanitize_url(url)

        if not is_valid_url(url):
            return CrawlResult(
                url=url,
                success=False,
                error_message="Invalid or disallowed URL",
            )

        if not bypass_cache:
            cached_result = await self._get_from_cache(url)
            if cached_result:
                return cached_result

        start_time = time.time()

        browser_config_dict = self.config.to_browser_config()
        if use_proxy and self.config.proxy_server:
            browser_config_dict["proxy"] = {"server": self.config.proxy_server}

        browser_config = BrowserConfig(**browser_config_dict)

        run_config_dict = self.config.to_crawler_run_config()
        run_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(),
            **run_config_dict,
        )

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)

                crawl_time_ms = int((time.time() - start_time) * 1000)

                if result.success:
                    markdown = result.markdown.raw_markdown if result.markdown else None
                    word_count = len(markdown.split()) if markdown else 0

                    crawl_result = CrawlResult(
                        url=result.url,
                        success=True,
                        markdown=markdown,
                        html=result.html,
                        title=result.metadata.get("title") if result.metadata else None,
                        links=result.links.get("internal", []) if result.links else [],
                        crawl_time_ms=crawl_time_ms,
                        word_count=word_count,
                    )

                    await self._save_to_cache(url, crawl_result)

                    return crawl_result
                else:
                    return CrawlResult(
                        url=url,
                        success=False,
                        error_message=result.error_message or "Unknown error",
                        crawl_time_ms=crawl_time_ms,
                    )

        except asyncio.TimeoutError:
            return CrawlResult(
                url=url,
                success=False,
                error_message="Crawl timeout",
            )
        except Exception as e:
            return CrawlResult(
                url=url,
                success=False,
                error_message=str(e),
            )

    async def crawl_recursive(
        self,
        urls: list[str],
        max_depth: int = 1,
        max_pages: int = 10,
    ) -> list[CrawlResult]:
        """
        Recursively crawl internal links from starting URLs.

        Args:
            urls: Starting URLs
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl

        Returns:
            List of crawl results
        """
        visited = set()
        results: list[CrawlResult] = []

        def normalize_url(url: str) -> str:
            return urldefrag(url)[0]

        current_urls = set([normalize_url(u) for u in urls])

        browser_config_dict = self.config.to_browser_config()
        browser_config = BrowserConfig(**browser_config_dict)

        run_config_dict = self.config.to_crawler_run_config()
        run_config = CrawlerRunConfig(**run_config_dict)

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=self.config.memory_threshold,
            check_interval=1.0,
            max_session_permit=self.config.max_concurrent,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            for depth in range(max_depth):
                if len(results) >= max_pages:
                    break

                urls_to_crawl = [
                    normalize_url(url)
                    for url in current_urls
                    if normalize_url(url) not in visited
                ][: max_pages - len(results)]

                if not urls_to_crawl:
                    break

                crawl_results = await crawler.arun_many(
                    urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
                )

                next_level_urls = set()

                for result in crawl_results:
                    norm_url = normalize_url(result.url)
                    visited.add(norm_url)

                    if result.success:
                        markdown = (
                            result.markdown.raw_markdown if result.markdown else None
                        )
                        word_count = len(markdown.split()) if markdown else 0

                        crawl_result = CrawlResult(
                            url=result.url,
                            success=True,
                            markdown=markdown,
                            html=result.html,
                            title=result.metadata.get("title")
                            if result.metadata
                            else None,
                            links=result.links.get("internal", [])
                            if result.links
                            else [],
                            word_count=word_count,
                        )
                        results.append(crawl_result)

                        for link in (
                            result.links.get("internal", []) if result.links else []
                        ):
                            next_url = normalize_url(link.get("href", ""))
                            if next_url not in visited and is_valid_url(next_url):
                                next_level_urls.add(next_url)
                    else:
                        results.append(
                            CrawlResult(
                                url=result.url,
                                success=False,
                                error_message=result.error_message,
                            )
                        )

                current_urls = next_level_urls

        return results


async def get_crawler() -> WebCrawler:
    """Get a crawler instance."""
    return WebCrawler()
