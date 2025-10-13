"""Redis cache client and operations."""

import json
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential



logger = LoggerAdapter(__name__)

# Global Redis client instance
redis_client: Optional[redis.Redis] = None


async def connect_to_redis() -> None:
    """Create Redis connection."""
    global redis_client

    try:
        logger.info(
            "Connecting to Redis", host=settings.redis_host, port=settings.redis_port
        )

        redis_client = await redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
        )

        # Verify connection
        await redis_client.ping()

        logger.info("Successfully connected to Redis")

    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise


async def close_redis_connection() -> None:
    """Close Redis connection."""
    global redis_client

    if redis_client:
        logger.info("Closing Redis connection")
        await redis_client.close()
        redis_client = None
        logger.info("Redis connection closed")


def get_redis_client() -> redis.Redis:
    """Get Redis client instance."""
    if not redis_client:
        raise RuntimeError("Redis is not connected")
    return redis_client


class CacheManager:
    """Manager for cache operations."""

    def __init__(self, prefix: str = "langchain"):
        self.prefix = prefix

    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self.prefix}:{key}"

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        client = get_redis_client()
        full_key = self._make_key(key)

        try:
            value = await client.get(full_key)
            if value:
                return json.loads(value)
            return None
        except json.JSONDecodeError:
            return value
        except Exception as e:
            logger.error("Cache get error", key=full_key, error=str(e))
            return None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        client = get_redis_client()
        full_key = self._make_key(key)
        ttl = ttl or settings.cache_ttl

        try:
            serialized = json.dumps(value) if not isinstance(value, str) else value
            await client.setex(full_key, ttl, serialized)
            return True
        except Exception as e:
            logger.error("Cache set error", key=full_key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        client = get_redis_client()
        full_key = self._make_key(key)

        try:
            result = await client.delete(full_key)
            return bool(result)
        except Exception as e:
            logger.error("Cache delete error", key=full_key, error=str(e))
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        client = get_redis_client()
        full_pattern = self._make_key(pattern)

        try:
            keys = []
            async for key in client.scan_iter(full_pattern):
                keys.append(key)

            if keys:
                return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(
                "Cache delete pattern error", pattern=full_pattern, error=str(e)
            )
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        client = get_redis_client()
        full_key = self._make_key(key)

        try:
            return bool(await client.exists(full_key))
        except Exception as e:
            logger.error("Cache exists error", key=full_key, error=str(e))
            return False

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        client = get_redis_client()
        full_keys = [self._make_key(k) for k in keys]

        try:
            values = await client.mget(full_keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
            return result
        except Exception as e:
            logger.error("Cache get many error", error=str(e))
            return {}

    async def set_many(
        self, mapping: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        client = get_redis_client()
        ttl = ttl or settings.cache_ttl

        try:
            pipe = client.pipeline()
            for key, value in mapping.items():
                full_key = self._make_key(key)
                serialized = json.dumps(value) if not isinstance(value, str) else value
                pipe.setex(full_key, ttl, serialized)
            await pipe.execute()
            return True
        except Exception as e:
            logger.error("Cache set many error", error=str(e))
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter in cache."""
        client = get_redis_client()
        full_key = self._make_key(key)

        try:
            return await client.incr(full_key, amount)
        except Exception as e:
            logger.error("Cache increment error", key=full_key, error=str(e))
            return None

    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for a key."""
        client = get_redis_client()
        full_key = self._make_key(key)

        try:
            ttl = await client.ttl(full_key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error("Cache get TTL error", key=full_key, error=str(e))
            return None


# Create global cache manager instance
redis_cache = CacheManager()
