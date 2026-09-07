"""Crawler feature service."""

import asyncio
import time
from typing import Any

from redis.asyncio import Redis
from returns.result import Failure, Success

from app.shared.crawler import (
    CrawlResult,
    GeminiProcessor,
    WebCrawler,
    smart_chunk_markdown,
    truncate_content,
)
from app.shared.crawler import SchemaType as ProcessorSchemaType
from app.shared.services import RateLimiter, RateLimitScope, search
from app.shared.services.errors import TavilyValidationError
from app.utils import logger, trace_layer

from .constants import CrawlMode
from .dto import (
    CrawlChunk,
    CrawlRequest,
    CrawlResponse,
    CrawlResultItem,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from .errors import CrawlerCrawlError, CrawlerResult, CrawlerSearchError, CrawlerValidationError


class CrawlerService:
    """Service for web crawling and searching."""

    def __init__(
        self,
        crawler: WebCrawler | None = None,
        processor: GeminiProcessor | None = None,
        rate_limiter: RateLimiter | None = None,
        redis_client: Redis | None = None,
    ):
        self._crawler = crawler
        self._processor = processor
        self._rate_limiter = rate_limiter
        self._redis_client = redis_client

    @property
    def crawler(self) -> WebCrawler:
        if self._crawler is None:
            message = "CrawlerService requires an injected crawler"
            raise RuntimeError(message)
        return self._crawler

    @property
    def processor(self) -> GeminiProcessor:
        if self._processor is None:
            message = "CrawlerService requires an injected processor"
            raise RuntimeError(message)
        return self._processor

    @property
    def rate_limiter(self) -> RateLimiter:
        if self._rate_limiter is None:
            message = "CrawlerService requires an injected rate limiter"
            raise RuntimeError(message)
        return self._rate_limiter

    @trace_layer("service")
    async def check_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if rate limit is exceeded."""
        return await self.rate_limiter.check_rate_limit(identifier, scope)

    @trace_layer("service")
    async def increment_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
    ) -> None:
        """Increment rate limit counter."""
        await self.rate_limiter.increment_rate_limit(identifier, scope)

    @trace_layer("service")
    async def crawl(self, request: CrawlRequest) -> CrawlerResult[CrawlResponse]:
        """
        Crawl a URL or URLs based on request.

        Args:
            request: Crawl request parameters

        Returns:
            CrawlResponse with results
        """
        start_time = time.time()

        logger.bind(url=request.url).info("Starting crawl")

        try:
            async with asyncio.timeout(request.timeout):
                if request.max_depth > 1:
                    crawl_result = await self.crawler.crawl_recursive(
                        urls=[request.url],
                        max_depth=request.max_depth,
                        max_pages=request.max_pages,
                        use_proxy=request.use_proxy,
                        bypass_cache=request.bypass_cache,
                    )
                else:
                    crawl_result = await self.crawler.crawl(
                        url=request.url,
                        use_proxy=request.use_proxy,
                        bypass_cache=request.bypass_cache,
                    )
        except TimeoutError:
            return Failure(CrawlerCrawlError(message="Crawl operation timed out"))
        if isinstance(crawl_result, Failure):
            error = crawl_result.failure()
            return Failure(CrawlerCrawlError(message=error.message, details=error.details))
        resolved = crawl_result.unwrap()
        crawl_results = resolved if isinstance(resolved, list) else [resolved]

        results: list[CrawlResultItem] = []
        total_word_count = 0
        successful_pages = 0
        failed_pages = 0

        remaining_output = request.max_total_output_chars
        for crawl_result in crawl_results:
            item = await self._process_crawl_result(
                crawl_result,
                request,
                output_budget=max(0, min(request.max_output_chars, remaining_output)),
            )
            results.append(item)
            remaining_output = max(0, remaining_output - len(item.markdown or ""))

            if item.success:
                successful_pages += 1
                total_word_count += item.word_count or 0
            else:
                failed_pages += 1

        processing_time_ms = int((time.time() - start_time) * 1000)

        return Success(
            CrawlResponse(
                success=failed_pages == 0,
                query_url=request.url,
                results=results,
                total_pages=len(results),
                successful_pages=successful_pages,
                failed_pages=failed_pages,
                total_word_count=total_word_count,
                processing_time_ms=processing_time_ms,
            )
        )

    async def _process_crawl_result(  # noqa: PLR0912
        self,
        crawl_result: CrawlResult,
        request: CrawlRequest,
        output_budget: int,
    ) -> CrawlResultItem:
        """Process a single crawl result with optional Gemini processing."""

        markdown = crawl_result.markdown
        extracted_data: dict[str, Any] | None = None
        summary: str | None = None
        content_truncated = False

        if crawl_result.success and markdown:
            content_truncated = len(markdown) > output_budget
            markdown = truncate_content(markdown, max_length=output_budget) if output_budget else ""

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

        links = [link.get("href", "") for link in (crawl_result.links or [])][: request.max_links]
        if content is None:
            content = ""
        if output_budget == 0 and content:
            content = ""
            content_truncated = True
        elif len(content) > output_budget:
            content = truncate_content(content, max_length=output_budget)
            content_truncated = True

        chunks = []
        if request.include_chunks and content:
            chunks = [
                CrawlChunk(**chunk.model_dump())
                for chunk in smart_chunk_markdown(content, max_len=request.chunk_size)[
                    : request.max_chunks
                ]
            ]

        return CrawlResultItem(
            url=crawl_result.url,
            success=crawl_result.success,
            title=crawl_result.title,
            markdown=content,
            html=content if request.mode == CrawlMode.HTML else None,
            summary=summary,
            extracted_data=extracted_data,
            word_count=crawl_result.word_count,
            crawl_time_ms=crawl_result.crawl_time_ms,
            cached=crawl_result.cached,
            error_message=crawl_result.error_message,
            links=links,
            content_truncated=content_truncated,
            chunks=chunks,
        )

    @staticmethod
    @trace_layer("service")
    async def search(request: SearchRequest) -> CrawlerResult[SearchResponse]:
        """
        Search the web using Tavily.

        Args:
            request: Search request parameters

        Returns:
            SearchResponse with results
        """
        logger.bind(query=request.query).info("Searching")

        tavily_result = await search(
            query=request.query,
            max_results=request.max_results,
            include_answer=request.include_answer,
        )
        if isinstance(tavily_result, Failure):
            error = tavily_result.failure()
            crawler_error = (
                CrawlerValidationError(message=error.message, details=error.details)
                if isinstance(error, TavilyValidationError)
                else CrawlerSearchError(message=error.message, details=error.details)
            )
            return Failure(crawler_error)
        tavily_response = tavily_result.unwrap()

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

        return Success(
            SearchResponse(
                success=True,
                query=request.query,
                answer=tavily_response.answer,
                results=results,
                total_results=tavily_response.total_results,
            )
        )

    @trace_layer("service")
    async def close(self) -> None:
        """Close all connections."""
        if self._rate_limiter:
            await self._rate_limiter.close()  # ty: ignore[unresolved-attribute]
