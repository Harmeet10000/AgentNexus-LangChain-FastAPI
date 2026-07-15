"""Core crawler module using Crawl4AI."""

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

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
from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import get_settings
from app.utils import logger

from .config import CrawlerConfig, get_crawler_config
from .validator import is_valid_url, sanitize_url

if TYPE_CHECKING:
    from app.config.settings import Settings


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
    ):
        self.config = config or get_crawler_config()
        self.redis_client = redis_client

    @staticmethod
    def _get_cache_key(url: str) -> str:
        """Generate cache key for URL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
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
        except RedisError as exc:
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
        except RedisError as exc:
            exc.add_note(f"url={url}, operation=cache_write")
            logger.bind(operation="cache_write", url=url).exception("Cache write failed")

    @staticmethod
    def _to_crawl_result(result: Any, start_time: float) -> CrawlResult:
        """Convert crawl4ai result to domain CrawlResult."""
        crawl_time_ms = int((time.time() - start_time) * 1000)

        if result.success:
            markdown = result.markdown.raw_markdown if result.markdown else None
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
    ) -> CrawlResult:
        """Crawl a single URL."""
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
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)

                crawl_result = self._to_crawl_result(result, start_time)

                if crawl_result.success:
                    await self._save_to_cache(url, crawl_result)

                return crawl_result

        except TimeoutError:
            return CrawlResult(
                url=url,
                success=False,
                error_message="Crawl timeout",
            )
        except (httpx.HTTPError, PlaywrightError) as e:
            e.add_note(f"url={url}")
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
        """Recursively crawl internal links using native BFS deep crawl strategy."""
        start_time: int | float = time.time()

        browser_config = BrowserConfig(**self.config.to_browser_config())

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

        # Auto-detect PDF URLs in seed list
        has_pdfs = any(self._is_pdf_url(u) for u in urls)
        if has_pdfs:
            run_config_dict["scraping_strategy"] = PDFContentScrapingStrategy()

        run_config = CrawlerRunConfig(
            deep_crawl_strategy=deep_crawl,
            markdown_generator=md_generator,
            **run_config_dict,
        )

        dispatcher = self._build_dispatcher()

        async with AsyncWebCrawler(config=browser_config) as crawler:
            crawl_results = await crawler.arun_many(
                urls=urls,
                config=run_config,
                dispatcher=dispatcher,
            )

        return [self._to_crawl_result(result, start_time) for result in crawl_results]
