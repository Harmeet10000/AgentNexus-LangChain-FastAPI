"""Redis caching example using the project's Result and logging conventions.

Run with ``uv run python -m app.examples.redis_examples`` to exercise the
service flow without requiring Redis. The router below is illustrative: real
Redis calls are translated at the adapter boundary, services return typed
``Result`` values, and routers render them with ``render_result``.

Redis data types:

- Strings: simple JSON values and cache-aside lookups.
- Hashes: object-like values with partial field updates.
- Lists: ordered activity feeds and notifications.
- Pipelines: several commands in one round trip.
- Search indexes and Bloom filters: optional Redis modules.

All cache keys use ``{object_type}:{key}``, and cache entries should have a TTL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated  # noqa: TC003

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, EmailStr
from redis.asyncio import Redis  # noqa: TC002
from returns.result import Failure, Success

from app.connections.redis import get_redis
from app.shared.result import (  # noqa: TC001
    APIResponse,
    render_result,
)
from app.utils import logger
from app.utils.cache import (
    delete_cache,
    delete_list,
    get_cache,
    get_list_items,
    push_to_list,
    set_cache,
)

if TYPE_CHECKING:
    from app.utils.cache import CacheResult


class UserProfile(BaseModel):
    """User profile data."""

    user_id: str
    name: str
    email: EmailStr
    phone: str | None = None
    bio: str | None = None
    avatar_url: str | None = None


class UserUpdate(BaseModel):
    """Partial user update."""

    name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    bio: str | None = None


class ActivityLog(BaseModel):
    """Activity log entry."""

    timestamp: str
    action: str
    resource: str
    details: dict[str, object] | None = None


async def get_user_from_db(user_id: str) -> UserProfile:
    """Simulate a database lookup used by the cache-aside example."""
    logger.bind(user_id=user_id, operation="get_user_from_db").info("Database lookup")
    return UserProfile(
        user_id=user_id,
        name="John Doe",
        email="john@example.com",
        phone="123-456-7890",
    )


async def get_cached_user(user_id: str, redis: Redis) -> CacheResult[UserProfile]:
    """Read through Redis, falling back to the database when cache access degrades."""
    cached_result = await get_cache(redis, "user", user_id)
    if isinstance(cached_result, Failure):
        logger.bind(user_id=user_id, operation="get_cache").warning("Cache read degraded")
        cached = None
    else:
        cached = cached_result.unwrap()

    if cached:
        logger.bind(user_id=user_id, cache_hit=True).info("Cache hit")
        return Success(UserProfile(**cached))

    logger.bind(user_id=user_id, cache_hit=False).info("Cache miss")
    user = await get_user_from_db(user_id)

    set_result = await set_cache(
        redis,
        "user",
        user_id,
        user.model_dump(),
        expire_seconds=3600,
    )
    if isinstance(set_result, Failure):
        logger.bind(user_id=user_id, operation="set_cache").warning("Cache write degraded")
    return Success(user)


async def update_cached_user(
    user_id: str,
    updates: UserUpdate,
    redis: Redis,
) -> CacheResult[UserProfile]:
    """Update the simulated database and refresh the cache when possible."""
    user = await get_user_from_db(user_id)
    updated = user.model_copy(update=updates.model_dump(exclude_none=True))
    result = await set_cache(
        redis,
        "user",
        user_id,
        updated.model_dump(),
        expire_seconds=3600,
    )
    if isinstance(result, Failure):
        logger.bind(user_id=user_id, operation="set_cache", error=result.failure()).warning(
            "Cache refresh degraded"
        )
    return Success(updated)


async def delete_user_cache(user_id: str, redis: Redis) -> CacheResult[dict[str, str]]:
    """Invalidate the profile and activity caches."""
    profile_result = await delete_cache(redis, "user", user_id)
    if isinstance(profile_result, Failure):
        return profile_result
    activity_result = await delete_list(redis, "activity", user_id)
    if isinstance(activity_result, Failure):
        return activity_result
    return Success({"status": "deleted", "user_id": user_id})


async def add_activity(
    user_id: str, activity: ActivityLog, redis: Redis
) -> CacheResult[dict[str, str]]:
    """Prepend an activity and retain the latest 30 days through the cache TTL."""
    result = await push_to_list(
        redis,
        "activity",
        user_id,
        activity.model_dump(),
        prepend=True,
        expire_seconds=2592000,
    )
    if isinstance(result, Failure):
        return result
    return Success({"status": "logged", "user_id": user_id, "action": activity.action})


async def get_user_activity(
    user_id: str,
    redis: Redis,
    limit: int = 20,
) -> CacheResult[list[ActivityLog]]:
    """Read the most recent activity entries from a Redis list."""
    result = await get_list_items(redis, "activity", user_id, start=0, end=limit - 1)
    if isinstance(result, Failure):
        return result
    return Success([ActivityLog(**item) for item in result.unwrap()])


router = APIRouter(prefix="/api/v1/users", tags=["Users"])


@router.get("/{user_id}")
async def get_user_endpoint(
    user_id: str,
    response: Response,
    redis: Annotated[Redis, Depends(get_redis)],
) -> APIResponse[UserProfile]:
    """Get a user through cache-aside lookup and render the Result."""
    result = await get_cached_user(user_id, redis)
    return render_result(result, response, message="User retrieved")


@router.patch("/{user_id}")
async def update_user_endpoint(
    user_id: str,
    updates: UserUpdate,
    response: Response,
    redis: Annotated[Redis, Depends(get_redis)],
) -> APIResponse[UserProfile]:
    """Update a user and refresh its cache."""
    result = await update_cached_user(user_id, updates, redis)
    return render_result(result, response, message="User updated")


@router.delete("/{user_id}")
async def delete_user_endpoint(
    user_id: str,
    response: Response,
    redis: Annotated[Redis, Depends(get_redis)],
) -> APIResponse[dict[str, str]]:
    """Invalidate all cache entries for a user."""
    result = await delete_user_cache(user_id, redis)
    return render_result(result, response, message="User cache invalidated")


@router.post("/{user_id}/activity")
async def log_activity_endpoint(
    user_id: str,
    activity: ActivityLog,
    response: Response,
    redis: Annotated[Redis, Depends(get_redis)],
) -> APIResponse[dict[str, str]]:
    """Append an activity to the user's Redis list."""
    result = await add_activity(user_id, activity, redis)
    return render_result(result, response, message="Activity logged")


@router.get("/{user_id}/activity")
async def get_activity_endpoint(
    user_id: str,
    response: Response,
    redis: Annotated[Redis, Depends(get_redis)],
    limit: int = 20,
) -> APIResponse[list[ActivityLog]]:
    """Read recent activity entries from Redis."""
    result = await get_user_activity(user_id, redis, limit)
    return render_result(result, response, message="Activity retrieved")


async def _demo() -> None:
    """Demonstrate the service result shape without connecting to Redis."""
    logger.info("Redis example loaded; run the router with a configured Redis client")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo())
