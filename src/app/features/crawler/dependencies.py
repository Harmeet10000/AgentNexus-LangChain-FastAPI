"""Crawler feature dependencies."""

from typing import TYPE_CHECKING

from fastapi import Request

from app.features.crawler.service import CrawlerService
from app.shared.services.rate_limiter import RateLimiter

if TYPE_CHECKING:
    from redis.asyncio import Redis


async def get_crawler_service(request: Request) -> CrawlerService:
    """Get crawler service instance with Redis from app.state."""
    redis_client: Redis | None = getattr(request.app.state, "redis", None)
    return CrawlerService(redis_client=redis_client)


async def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter instance with Redis from app.state."""
    from app.shared.services import get_rate_limiter

    redis_client: Redis | None = getattr(request.app.state, "redis", None)
    return get_rate_limiter(redis_client=redis_client)
