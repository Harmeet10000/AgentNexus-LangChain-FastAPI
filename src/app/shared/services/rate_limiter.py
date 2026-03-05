"""Rate limiting using fastapi-limiter and Redis."""

from enum import Enum

from pydantic import BaseModel
from redis.asyncio import Redis

from app.config import get_settings


class RateLimitScope(str, Enum):
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


class RateLimiter:
    """Rate limiter using fastapi-limiter and Redis backend."""

    def __init__(self, redis_client: Redis | None = None):
        self.redis_client = redis_client

    async def check_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
    ) -> tuple[bool, dict]:
        """
        Check if rate limit is exceeded.

        Args:
            identifier: User identifier (user_id or IP)
            scope: Rate limit scope

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        if not self.redis_client:
            return True, {}

        config = RATE_LIMIT_CONFIGS.get(scope)
        if not config:
            return True, {}

        minute_key = f"rate:{scope.value}:{identifier}:min"
        hour_key = f"rate:{scope.value}:{identifier}:hour"

        minute_count = await self.redis_client.get(minute_key)
        hour_count = await self.redis_client.get(hour_key)

        current_minute = int(minute_count) if minute_count else 0
        current_hour = int(hour_count) if hour_count else 0

        if current_minute >= config.per_minute:
            ttl_minute = await self.redis_client.ttl(minute_key)
            return False, {
                "error": "Rate limit exceeded",
                "limit": config.per_minute,
                "window": "minute",
                "retry_after": ttl_minute if ttl_minute > 0 else 60,
            }

        if current_hour >= config.per_hour:
            ttl_hour = await self.redis_client.ttl(hour_key)
            return False, {
                "error": "Rate limit exceeded",
                "limit": config.per_hour,
                "window": "hour",
                "retry_after": ttl_hour if ttl_hour > 0 else 3600,
            }

        return True, {
            "remaining_minute": config.per_minute - current_minute - 1,
            "remaining_hour": config.per_hour - current_hour - 1,
        }

    async def increment_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
    ):
        """Increment rate limit counter."""
        if not self.redis_client:
            return

        minute_key = f"rate:{scope.value}:{identifier}:min"
        hour_key = f"rate:{scope.value}:{identifier}:hour"

        pipe = self.redis_client.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)
        await pipe.execute()

    async def get_remaining(
        self,
        identifier: str,
        scope: RateLimitScope,
    ) -> dict:
        """Get remaining rate limit quota."""
        if not self.redis_client:
            return {"remaining_minute": 0, "remaining_hour": 0}

        config = RATE_LIMIT_CONFIGS.get(scope)
        if not config:
            return {"remaining_minute": 0, "remaining_hour": 0}

        minute_key = f"rate:{scope.value}:{identifier}:min"
        hour_key = f"rate:{scope.value}:{identifier}:hour"

        minute_count = await self.redis_client.get(minute_key)
        hour_count = await self.redis_client.get(hour_key)

        current_minute = int(minute_count) if minute_count else 0
        current_hour = int(hour_count) if hour_count else 0

        return {
            "remaining_minute": max(0, config.per_minute - current_minute),
            "remaining_hour": max(0, config.per_hour - current_hour),
        }


_rate_limiter: RateLimiter | None = None


def get_rate_limiter(redis_client: Redis | None = None) -> RateLimiter:
    """Get rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(redis_client)
    elif redis_client is not None:
        _rate_limiter.redis_client = redis_client
    return _rate_limiter
