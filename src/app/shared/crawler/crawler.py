"""Core crawler module using Crawl4AI."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any, Protocol, cast  # noqa: TC003

import httpx
from crawl4ai import (
    AsyncUrlSeeder,
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerMonitor,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
    SeedingConfig,
)
from crawl4ai.async_dispatcher import RateLimiter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.processors.pdf import PDFContentScrapingStrategy
from playwright.async_api import Error as PlaywrightError
from pydantic import BaseModel, ConfigDict, ValidationError
from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.config import get_settings
from app.utils import logger

from .chunker import truncate_content
from .config import get_crawler_config
from .errors import CrawlerProviderError
from .validator import sanitize_url, validate_url_for_fetch

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from app.config.settings import Settings

    from .config import CrawlerConfig
    from .errors import CrawlerProcessingResult


class CrawlerBrowser(Protocol):
    """Minimum browser interface required by the domain crawler."""

    async def arun(self, *, url: str, config: CrawlerRunConfig) -> Any: ...

    async def arun_many(self, *, urls: list[str], config: CrawlerRunConfig, dispatcher: Any) -> Any: ...


class CrawlResult(BaseModel):
    """Result from crawling a URL."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    """Web crawler using Crawl4AI with caching."""

    def __init__(
        self,
        config: CrawlerConfig | None = None,
        redis_client: Redis | None = None,
        browser: CrawlerBrowser | None = None,
    ):
        self.config = config or get_crawler_config()
        self.redis_client = redis_client
        self.browser = browser

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key that changes when content policy changes."""
        cache_material = "|".join(
            (
                "v2",
                url,
                self.config.user_agent,
                str(self.config.word_count_threshold),
                str(self.config.pruning_threshold),
                ",".join(self.config.excluded_tags),
                str(self.config.max_content_size),
            )
        )
        url_hash = hashlib.sha256(cache_material.encode()).hexdigest()[:16]
        return f"crawl:cache:{url_hash}"

    async def _get_from_cache(self, url: str) -> CrawlResult | None:
        """Get cached crawl result."""
        if not self.redis_client:
            return None

        try:
            cache_key = self._get_cache_key(url)
            cached = await self.redis_client.get(cache_key)

            if cached:
                data = json.loads(cached)
                return CrawlResult(
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
        except (RedisError, json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
            exc.add_note(f"url={url}, operation=cache_read")
            logger.bind(operation="cache_read", url=url).exception("Cache read failed")
        return None

    async def _save_to_cache(self, url: str, result: CrawlResult) -> None:
        """Save crawl result to cache."""
        if not self.redis_client:
            return

        try:
            settings: Settings = get_settings()
            cache_key = self._get_cache_key(url)

            data: dict[str, str | int | list[dict[str, Any]] | None] = {
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

            await self.redis_client.setex(
                cache_key,
                settings.REDIS_CRAWL_CACHE_TTL,
                json.dumps(data),
            )
        except (RedisError, TypeError, ValueError) as exc:
            exc.add_note(f"url={url}, operation=cache_write")
            logger.bind(operation="cache_write", url=url).exception("Cache write failed")

    @staticmethod
    def _to_crawl_result(result: Any, start_time: float) -> CrawlResult:
        """Convert crawl4ai result to domain CrawlResult."""
        crawl_time_ms = int((time.time() - start_time) * 1000)

        if result.success:
            markdown = None
            if result.markdown:
                markdown = result.markdown.fit_markdown or result.markdown.raw_markdown
            word_count = len(markdown.split()) if markdown else 0
            return CrawlResult(
                url=result.url,
                success=True,
                markdown=markdown,
                html=result.html,
                title=result.metadata.get("title") if result.metadata else None,
                links=result.links.get("internal", []) if result.links else [],
                crawl_time_ms=crawl_time_ms,
                word_count=word_count,
            )

        return CrawlResult(
            url=result.url,
            success=False,
            error_message=result.error_message or "Unknown error",
            crawl_time_ms=crawl_time_ms,
        )

    @staticmethod
    async def discover_urls(
        domain: str,
        pattern: str | None = None,
        max_urls: int = 50,
    ) -> list[str]:
        """Discover URLs for a domain via sitemap without full crawl."""
        async with AsyncUrlSeeder() as seeder:
            seeded = await seeder.urls(
                domain,
                SeedingConfig(
                    source="sitemap",
                    pattern=pattern or f"*{domain}*",
                    max_urls=max_urls,
                    extract_head=False,
                ),
            )
        return [row["url"] for row in seeded if row.get("status") == "valid"]

    @staticmethod
    def _is_pdf_url(url: str) -> bool:
        """Check if URL points to a PDF."""
        return url.lower().rstrip("/").endswith(".pdf")

    def _build_dispatcher(self) -> MemoryAdaptiveDispatcher:
        """Build dispatcher with optional CrawlerMonitor."""
        kwargs: dict[str, Any] = {
            "memory_threshold_percent": self.config.memory_threshold,
            "max_session_permit": self.config.max_concurrent,
            "rate_limiter": RateLimiter(
                base_delay=self.config.rate_limit_delay,
                max_retries=2,
            ),
        }
        if self.config.enable_monitor:
            kwargs["monitor"] = CrawlerMonitor()
        return MemoryAdaptiveDispatcher(**kwargs)

    async def crawl(
        self,
        url: str,
        use_proxy: bool = False,
        bypass_cache: bool = False,
    ) -> CrawlerProcessingResult[CrawlResult]:
        """Crawl a single URL."""
        url = sanitize_url(url)

        valid, validation_error = await validate_url_for_fetch(url)
        if not valid:
            return Failure(
                CrawlerProviderError(
                    message=validation_error or "Invalid or disallowed URL",
                    url=url,
                )
            )

        if not bypass_cache:
            cached_result = await self._get_from_cache(url)
            if cached_result:
                return Success(cached_result)

        start_time = time.time()

        browser_config_dict = self.config.to_browser_config()
        if use_proxy and self.config.proxy_server:
            browser_config_dict["proxy"] = {"server": self.config.proxy_server}

        browser_config = BrowserConfig(**browser_config_dict)

        # SPEC-05: Use MarkdownGenerator with content filters
        md_generator = self.config.get_markdown_generator()
        run_config_dict = self.config.to_crawler_run_config()

        # Auto-detect PDF URLs
        if self._is_pdf_url(url):
            run_config_dict["scraping_strategy"] = PDFContentScrapingStrategy()

        run_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            **run_config_dict,
        )

        try:
            if self.browser is not None:
                result = await self.browser.arun(url=url, config=run_config)
                return await self._finish_single_result(url, result, start_time)

            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)
                return await self._finish_single_result(url, result, start_time)

        except TimeoutError:
            return Failure(
                CrawlerProviderError(
                    message="Crawl timeout",
                    url=url,
                )
            )
        except (httpx.HTTPError, PlaywrightError) as e:
            e.add_note(f"url={url}")
            return Failure(
                CrawlerProviderError(
                    message=str(e),
                    url=url,
                )
            )

    async def _finish_single_result(
        self,
        url: str,
        result: Any,
        start_time: float,
    ) -> CrawlerProcessingResult[CrawlResult]:
        crawl_result = self._bound_result(self._to_crawl_result(result, start_time))
        if crawl_result.success:
            await self._save_to_cache(url, crawl_result)
        if not crawl_result.success:
            return Failure(
                CrawlerProviderError(
                    message=crawl_result.error_message or "Crawler provider failed",
                    url=url,
                )
            )
        return Success(crawl_result)

    async def crawl_recursive(
        self,
        urls: list[str],
        max_depth: int = 1,
        max_pages: int = 10,
        use_proxy: bool = False,
        bypass_cache: bool = False,
    ) -> CrawlerProcessingResult[list[CrawlResult]]:
        """Recursively crawl internal links using native BFS deep crawl strategy."""
        if not urls:
            return Failure(
                CrawlerProviderError(
                    message="At least one URL is required",
                    url="",
                )
            )

        normalized_urls: list[str] = []
        for raw_url in urls:
            url = sanitize_url(raw_url)
            valid, validation_error = await validate_url_for_fetch(url)
            if not valid:
                return Failure(
                    CrawlerProviderError(
                        message=validation_error or "Invalid or disallowed URL",
                        url=url,
                    )
                )
            normalized_urls.append(url)

        start_time: int | float = time.time()

        browser_config_dict = self.config.to_browser_config()
        if use_proxy and self.config.proxy_server:
            browser_config_dict["proxy"] = {"server": self.config.proxy_server}
        browser_config = BrowserConfig(**browser_config_dict)

        # SPEC-06: Use native BFSDeepCrawlStrategy
        deep_crawl = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=False,
        )

        # SPEC-05: MarkdownGenerator with content filters
        md_generator = self.config.get_markdown_generator()

        # SPEC-07: Rate limiter on dispatcher
        run_config_dict = self.config.to_crawler_run_config()
        run_config_dict["cache_mode"] = "bypass" if bypass_cache else self.config.cache_mode

        # Auto-detect PDF URLs in seed list
        has_pdfs = any(self._is_pdf_url(u) for u in normalized_urls)
        if has_pdfs:
            run_config_dict["scraping_strategy"] = PDFContentScrapingStrategy()

        run_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_crawl,
            markdown_generator=md_generator,
            **run_config_dict,
        )

        dispatcher = self._build_dispatcher()

        try:
            if self.browser is not None:
                crawl_results = await self.browser.arun_many(
                    urls=normalized_urls,
                    config=run_config,
                    dispatcher=dispatcher,
                )
            else:
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    crawl_results = await crawler.arun_many(
                        urls=normalized_urls,
                        config=run_config,
                        dispatcher=dispatcher,
                    )
        except (TimeoutError, httpx.HTTPError, PlaywrightError) as exc:
            return Failure(
                CrawlerProviderError(
                    message=str(exc),
                    url=normalized_urls[0],
                )
            )

        results = [
            self._bound_result(self._to_crawl_result(result, start_time))
            for result in cast("list[Any]", crawl_results)
        ]
        return Success(results)

    def _bound_result(self, result: CrawlResult) -> CrawlResult:
        """Apply the process-level content and link limits before caching."""
        max_size = self.config.max_content_size
        return result.model_copy(
            update={
                "markdown": truncate_content(result.markdown, max_size)
                if result.markdown is not None
                else None,
                "html": truncate_content(result.html, max_size)
                if result.html is not None
                else None,
                "links": (result.links or [])[:1_000],
            }
        )
