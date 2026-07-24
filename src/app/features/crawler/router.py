"""Crawler feature API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.shared.services import RateLimitScope
from app.shared.services.rate_limiter import RateLimiter
from app.utils import TooManyRequestsException

from .constants import CRAWLER_PREFIX, CRAWLER_TAG
from .dependencies import get_crawler_service, get_rate_limiter
from .dto import (
    CrawlRequest,
    CrawlResponse,
    RateLimitInfo,
    SearchRequest,
    SearchResponse,
)
from .service import CrawlerService

router = APIRouter(prefix=CRAWLER_PREFIX, tags=[CRAWLER_TAG])


def get_client_identifier(request: Request) -> str:
    """Get client identifier from request (IP or user ID)."""
    if hasattr(request.state, "user_id"):
        return str(request.state.user_id)

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"


@router.post(path="/crawl")
async def crawl_url(
    request_data: CrawlRequest,
    request: Request,
    service: Annotated[CrawlerService, Depends(get_crawler_service)],
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> CrawlResponse:
    """
    Crawl a URL and optionally extract structured data.

    - **url**: URL to crawl
    - **mode**: Output mode (markdown, html, text, summary)
    - **max_depth**: Recursion depth (1 = single page, 2+ = follow internal links)
    - **max_pages**: Maximum pages to crawl for recursive crawl
    - **use_proxy**: Use proxy for crawling
    - **bypass_cache**: Bypass cached results
    - **extract_structured**: Extract structured data using Gemini
    - **schema_type**: Predefined schema type (product, article, person, job)
    - **custom_schema**: Custom JSON schema for extraction
    - **summary**: Generate summary using Gemini
    - **timeout**: Timeout in seconds
    """
    client_id = get_client_identifier(request)

    is_allowed, rate_info = await rate_limiter.check_rate_limit(client_id, RateLimitScope.CRAWL)
    if not is_allowed:
        raise TooManyRequestsException(
            detail=rate_info.get("error") or "Rate limit exceeded",
            data={"retry_after": rate_info.get("retry_after")},
        )

    await rate_limiter.increment_rate_limit(client_id, RateLimitScope.CRAWL)

    return await service.crawl(request_data)


@router.get(path="/search")
async def search_web(
    request: Request,
    service: Annotated[CrawlerService, Depends(get_crawler_service)],
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    query: str,
    *,
    max_results: int = 10,
    include_answer: bool = True,
) -> SearchResponse:
    """
    Search the web using Tavily.

    - **query**: Search query
    - **max_results**: Maximum number of results (max 20)
    - **include_answer**: Include AI-generated answer
    """
    client_id = get_client_identifier(request)

    is_allowed, rate_info = await rate_limiter.check_rate_limit(client_id, RateLimitScope.SEARCH)
    if not is_allowed:
        raise TooManyRequestsException(
            detail=rate_info.get("error") or "Rate limit exceeded",
            data={"retry_after": rate_info.get("retry_after")},
        )

    await rate_limiter.increment_rate_limit(client_id, RateLimitScope.SEARCH)

    search_request = SearchRequest(
        query=query,
        max_results=max_results,
        include_answer=include_answer,
    )

    return await service.search(search_request)


@router.get(path="/rate-limit")
async def get_rate_limit_info(
    request: Request,
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> RateLimitInfo:
    """Get current rate limit information."""
    client_id = get_client_identifier(request)

    crawl_remaining = await rate_limiter.get_remaining(client_id, RateLimitScope.CRAWL)
    search_remaining = await rate_limiter.get_remaining(client_id, RateLimitScope.SEARCH)

    return RateLimitInfo(
        remaining_minute=min(
            crawl_remaining["remaining_minute"], search_remaining["remaining_minute"]
        ),
        remaining_hour=min(crawl_remaining["remaining_hour"], search_remaining["remaining_hour"]),
    )
