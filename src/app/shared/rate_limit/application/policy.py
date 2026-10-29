"""Rate limit policy: scopes, window configuration, per-provider limits.

This is the inward face of ``app.shared.rate_limit`` — the quota semantics
callers agree on, free of any Redis or transport binding. Concrete limiters
live under ``app.shared.rate_limit.adapters``.
"""

from enum import StrEnum

from pydantic import BaseModel

from app.config import get_settings


class RateLimitScope(StrEnum):
    """Rate limit scope types."""

    CRAWL = "crawl"
    SEARCH = "search"


class RateLimitConfig(BaseModel):
    """Rate limit configuration."""

    per_minute: int
    per_hour: int


RATE_LIMIT_CONFIGS = {
    RateLimitScope.CRAWL: RateLimitConfig(
        per_minute=get_settings().CRAWL_RATE_LIMIT_PER_MINUTE,
        per_hour=get_settings().CRAWL_RATE_LIMIT_PER_HOUR,
    ),
    RateLimitScope.SEARCH: RateLimitConfig(
        per_minute=get_settings().SEARCH_RATE_LIMIT_PER_MINUTE,
        per_hour=get_settings().SEARCH_RATE_LIMIT_PER_HOUR,
    ),
}
