from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from redis.asyncio import Redis


class RedisProtocolAdapter:
    """Adapt a project Redis client to FastAPI Guard's Redis handler protocol."""

    def __init__(self, redis: Redis, redis_prefix: str = "fastapi_guard:") -> None:
        self._redis = redis
        self.config = type("GuardRedisConfig", (), {"redis_prefix": redis_prefix})()

    async def initialize(self) -> None:
        await self._redis.ping()

    async def initialize_agent(self, agent_handler: Any) -> None:
        _ = agent_handler

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[Redis]:
        yield self._redis

    async def get_key(self, namespace: str, key: str) -> Any:
        return await self._redis.get(f"{self.config.redis_prefix}{namespace}:{key}")

    async def set_key(
        self, namespace: str, key: str, value: Any, ttl: int | None = None
    ) -> bool | None:
        namespaced_key = f"{self.config.redis_prefix}{namespace}:{key}"
        if ttl is not None:
            return bool(await self._redis.setex(namespaced_key, ttl, value))
        return bool(await self._redis.set(namespaced_key, value))

    async def delete(self, namespace: str, key: str) -> int | None:
        return int(await self._redis.delete(f"{self.config.redis_prefix}{namespace}:{key}"))

    async def keys(self, pattern: str) -> list[str] | None:
        keys = await self._redis.keys(f"{self.config.redis_prefix}{pattern}")
        return [key.decode() if isinstance(key, bytes) else str(key) for key in keys]
