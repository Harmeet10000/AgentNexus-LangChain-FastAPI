"""Crawler feature dependencies."""

from typing import TYPE_CHECKING

from fastapi import Request

from app.features.crawler.service import CrawlerService
from app.shared.crawler import GeminiProcessor, WebCrawler
from app.shared.services.rate_limiter import RateLimiter
from app.utils import ServiceUnavailableException

from .job_store import CrawlJobStore

if TYPE_CHECKING:
    from redis.asyncio import Redis


async def get_crawler_service(request: Request) -> CrawlerService:
    """Get crawler service instance with Redis from app.state."""
    redis_client: Redis | None = getattr(request.app.state, "redis", None)
    crawler = WebCrawler(
        redis_client=redis_client,
        browser=getattr(request.app.state, "crawl4ai_crawler", None),
    )
    processor = getattr(request.app.state, "crawler_processor", None)
    if processor is None:
        processor = GeminiProcessor()
    rate_limiter = await get_rate_limiter(request)
    return CrawlerService(
        crawler=crawler,
        processor=processor,
        rate_limiter=rate_limiter,
        redis_client=redis_client,
    )


async def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter instance with Redis from app.state."""
    from app.shared.services.rate_limiter import get_rate_limiter as build_rate_limiter

    redis_client: Redis | None = getattr(request.app.state, "redis", None)
    return build_rate_limiter(redis_client=redis_client)


async def get_crawl_job_store(request: Request) -> CrawlJobStore:
    """Return the durable job store or fail closed when Redis is unavailable."""
    redis_client: Redis | None = getattr(request.app.state, "redis", None)
    if redis_client is None:
        message = "Durable crawler jobs require Redis"
        raise ServiceUnavailableException(message)
    return CrawlJobStore(redis_client)
