"""Crawler feature dependencies."""

from app.features.crawler.service import CrawlerService
from app.shared.services.rate_limiter import RateLimiter


async def get_crawler_service() -> CrawlerService:
    """Get crawler service instance."""
    return CrawlerService()


async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    from app.shared.services import get_rate_limiter

    return get_rate_limiter()
