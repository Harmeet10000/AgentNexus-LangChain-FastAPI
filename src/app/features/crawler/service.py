"""Crawler feature service."""

import time
from typing import Any

from app.features.crawler.constants import CrawlMode, SchemaType
from app.features.crawler.dto import (
    CrawlRequest,
    CrawlResponse,
    CrawlResultItem,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.shared.crawler import (
    CrawlResult,
    GeminiProcessor,
    WebCrawler,
    get_crawler,
    get_processor,
    truncate_content,
)
from app.shared.crawler.processor import SchemaType as ProcessorSchemaType
from app.shared.services import RateLimitScope, TavilyClient, get_tavily_client
from app.shared.services.rate_limiter import RateLimiter
from app.utils.logger import logger


class CrawlerService:
    """Service for web crawling and searching."""

    def __init__(
        self,
        crawler: WebCrawler | None = None,
        processor: GeminiProcessor | None = None,
        tavily: TavilyClient | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self._crawler = crawler
        self._processor = processor
        self._tavily = tavily
        self._rate_limiter = rate_limiter

    @property
    def crawler(self) -> WebCrawler:
        if self._crawler is None:
            import asyncio

            self._crawler = asyncio.run(get_crawler())
        return self._crawler

    @property
    def processor(self) -> GeminiProcessor:
        if self._processor is None:
            import asyncio

            self._processor = asyncio.run(get_processor())
        return self._processor

    @property
    def tavily(self) -> TavilyClient:
        if self._tavily is None:
            import asyncio

            self._tavily = asyncio.run(get_tavily_client())
        return self._tavily

    @property
    def rate_limiter(self) -> RateLimiter:
        if self._rate_limiter is None:
            import asyncio

            self._rate_limiter = asyncio.run(self._get_rate_limiter())
        return self._rate_limiter

    async def _get_rate_limiter(self) -> RateLimiter:
        from app.shared.services import get_rate_limiter

        return get_rate_limiter()

    async def check_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
    ) -> tuple[bool, dict]:
        """Check if rate limit is exceeded."""
        return await self.rate_limiter.check_rate_limit(identifier, scope)

    async def increment_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
    ):
        """Increment rate limit counter."""
        await self.rate_limiter.increment_rate_limit(identifier, scope)

    async def crawl(self, request: CrawlRequest) -> CrawlResponse:
        """
        Crawl a URL or URLs based on request.

        Args:
            request: Crawl request parameters

        Returns:
            CrawlResponse with results
        """
        start_time = time.time()

        logger.info(f"Starting crawl for: {request.url}")

        if request.max_depth > 1:
            crawl_results = await self.crawler.crawl_recursive(
                urls=[request.url],
                max_depth=request.max_depth,
                max_pages=request.max_pages,
            )
        else:
            result = await self.crawler.crawl(
                url=request.url,
                use_proxy=request.use_proxy,
                bypass_cache=request.bypass_cache,
            )
            crawl_results = [result]

        results: list[CrawlResultItem] = []
        total_word_count = 0
        successful_pages = 0
        failed_pages = 0

        for crawl_result in crawl_results:
            item = await self._process_crawl_result(
                crawl_result,
                request,
            )
            results.append(item)

            if item.success:
                successful_pages += 1
                total_word_count += item.word_count or 0
            else:
                failed_pages += 1

        processing_time_ms = int((time.time() - start_time) * 1000)

        return CrawlResponse(
            success=failed_pages == 0,
            query_url=request.url,
            results=results,
            total_pages=len(results),
            successful_pages=successful_pages,
            failed_pages=failed_pages,
            total_word_count=total_word_count,
            processing_time_ms=processing_time_ms,
        )

    async def _process_crawl_result(
        self,
        crawl_result: CrawlResult,
        request: CrawlRequest,
    ) -> CrawlResultItem:
        """Process a single crawl result with optional Gemini processing."""

        markdown = crawl_result.markdown
        extracted_data: dict[str, Any] | None = None
        summary: str | None = None

        if crawl_result.success and markdown:
            markdown = truncate_content(markdown)

            if request.extract_structured:
                schema_type = None
                if request.schema_type:
                    schema_type = ProcessorSchemaType(request.schema_type.value)

                extraction_result = await self.processor.extract_structured(
                    content=markdown,
                    schema_type=schema_type,
                    custom_schema=request.custom_schema,
                )

                if extraction_result.success:
                    extracted_data = extraction_result.extracted_data

            if request.summary:
                summary_result = await self.processor.summarize(markdown)
                if summary_result.success:
                    summary = summary_result.summary

        if request.mode == CrawlMode.HTML:
            content = crawl_result.html
        elif request.mode == CrawlMode.TEXT:
            content = crawl_result.markdown
        elif request.mode == CrawlMode.SUMMARY:
            content = summary or crawl_result.markdown
        else:
            content = crawl_result.markdown

        links = [link.get("href", "") for link in (crawl_result.links or [])]

        return CrawlResultItem(
            url=crawl_result.url,
            success=crawl_result.success,
            title=crawl_result.title,
            markdown=content,
            html=crawl_result.html if request.mode == CrawlMode.HTML else None,
            summary=summary,
            extracted_data=extracted_data,
            word_count=crawl_result.word_count,
            crawl_time_ms=crawl_result.crawl_time_ms,
            cached=crawl_result.cached,
            error_message=crawl_result.error_message,
            links=links,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Search the web using Tavily.

        Args:
            request: Search request parameters

        Returns:
            SearchResponse with results
        """
        logger.info(f"Searching for: {request.query}")

        tavily_response = await self.tavily.search(
            query=request.query,
            max_results=request.max_results,
            include_answer=request.include_answer,
        )

        results = [
            SearchResultItem(
                url=result.url,
                title=result.title,
                content=result.content,
                score=result.score,
                published_date=result.published_date,
            )
            for result in tavily_response.results
        ]

        return SearchResponse(
            success=True,
            query=request.query,
            answer=tavily_response.answer,
            results=results,
            total_results=tavily_response.total_results,
        )

    async def close(self):
        """Close all connections."""
        if self._crawler:
            await self._crawler.close()
        if self._rate_limiter:
            await self._rate_limiter.close()
