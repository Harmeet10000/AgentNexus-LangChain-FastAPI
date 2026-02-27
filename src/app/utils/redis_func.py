"""Redis cache utility functions."""

import json
from typing import Any, TypeVar

from redis.asyncio import Redis

from app.config.settings import get_settings

T = TypeVar("T")


class RedisCache:
    """Redis cache helper class."""

    def __init__(
        self, redis_client: Redis, prefix: str = "cache", ttl: int | None = None
    ):
        self.redis = redis_client
        self.prefix = prefix
        self.ttl = ttl or get_settings().REDIS_CRAWL_CACHE_TTL

    def _get_key(self, key: str) -> str:
        """Generate cache key with prefix."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get value from cache."""
        try:
            value = await self.redis.get(self._get_key(key))
            if value:
                return json.loads(value)
        except Exception:
            pass
        return None

    async def set(
        self, key: str, value: dict[str, Any], ttl: int | None = None
    ) -> bool:
        """Set value in cache with optional TTL override."""
        try:
            await self.redis.setex(
                self._get_key(key),
                ttl or self.ttl,
                json.dumps(value),
            )
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            await self.redis.delete(self._get_key(key))
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self.redis.exists(self._get_key(key)) > 0
        except Exception:
            return False

    async def increment(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        """Increment a counter and optionally set TTL."""
        try:
            full_key = self._get_key(key)
            result = await self.redis.incr(full_key, amount)
            if ttl:
                await self.redis.expire(full_key, ttl)
            return result
        except Exception:
            return 0

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key."""
        try:
            return await self.redis.ttl(self._get_key(key))
        except Exception:
            return -2


async def get_cache(
    redis_client: Redis, prefix: str = "cache", ttl: int | None = None
) -> RedisCache:
    """Get RedisCache instance."""
    return RedisCache(redis_client, prefix, ttl)
