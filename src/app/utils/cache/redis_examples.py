"""Example: Redis caching in FastAPI endpoints.

This module demonstrates how to use the Redis cache utilities
in real FastAPI endpoints with proper error handling and dependency injection.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from redis.asyncio import Redis

from app.connections.redis import get_redis
from app.utils.cache import (
    delete_cache,
    delete_list,
    get_cache,
    get_list_items,
    push_to_list,
    set_cache,
)
from app.utils.exceptions import DatabaseException
from app.utils.logger import logger

# ────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────


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
    details: dict | None = None


# ────────────────────────────────────────────────────────────
# Repository-like functions (simulated database)
# ────────────────────────────────────────────────────────────


async def get_user_from_db(user_id: str) -> UserProfile:
    """Simulate database lookup."""
    # In real app: return await db.get_user(user_id)
    logger.info(f"Database lookup: user {user_id}")
    return UserProfile(
        user_id=user_id,
        name="John Doe",
        email="john@example.com",
        phone="123-456-7890",
    )


# ────────────────────────────────────────────────────────────
# Service functions with caching
# ────────────────────────────────────────────────────────────


async def get_cached_user(user_id: str, redis: Redis) -> UserProfile:
    """Get user profile with caching strategy.

    1. Check cache first (fast)
    2. If miss, fetch from DB (slow)
    3. Update cache for next time
    """
    try:
        # Try cache
        cached = await get_cache(redis, "user", user_id)
        if cached:
            logger.info(f"Cache HIT: user:{user_id}")
            return UserProfile(**cached)

        logger.info(f"Cache MISS: user:{user_id}")

    except DatabaseException as e:
        logger.warning(f"Cache read failed for user {user_id}: {e.detail}")
        # Continue to DB lookup

    # Cache miss or error - fetch from DB
    user = await get_user_from_db(user_id)

    # Update cache for next request
    try:
        await set_cache(redis, "user", user_id, user.model_dump(), expire_seconds=3600)
        logger.info(f"Cached user: user:{user_id}")
    except DatabaseException as e:
        logger.error(f"Failed to cache user {user_id}: {e.detail}")
        # Don't fail the request, just continue without cache

    return user


async def update_cached_user(
    user_id: str, updates: UserUpdate, redis: Redis
) -> UserProfile:
    """Update user and refresh cache.

    For small updates, use Hash to avoid re-fetching entire document.
    """
    # Update DB (simulated)
    user = await get_user_from_db(user_id)
    update_data = updates.model_dump(exclude_none=True)
    user_dict = user.model_dump()
    user_dict.update(update_data)
    user = UserProfile(**user_dict)

    # Update cache
    try:
        # Option 1: Replace entire cache
        await set_cache(redis, "user", user_id, user.model_dump(), expire_seconds=3600)

        # Option 2: Update only changed fields (more efficient for Hash type)
        # await update_hash(redis, "user", user_id, update_data)

        logger.info(f"Updated cache for user: {user_id}")
    except DatabaseException as e:
        logger.error(f"Failed to update cache for user {user_id}: {e.detail}")

    return user


async def invalidate_user_cache(user_id: str, redis: Redis) -> None:
    """Invalidate user cache (use when user deleted or cache expires)."""
    try:
        deleted = await delete_cache(redis, "user", user_id)
        if deleted:
            logger.info(f"Invalidated cache: user:{user_id}")
    except DatabaseException as e:
        logger.error(f"Failed to invalidate cache for user {user_id}: {e.detail}")


async def add_activity(user_id: str, activity: ActivityLog, redis: Redis) -> None:
    """Add activity to user's activity feed."""
    try:
        await push_to_list(
            redis,
            "activity",
            user_id,
            activity.model_dump(),
            prepend=True,  # Most recent first
            expire_seconds=2592000,  # 30 days
        )
        logger.info(f"Activity logged for user: {user_id}")
    except DatabaseException as e:
        logger.error(f"Failed to log activity for user {user_id}: {e.detail}")


async def get_user_activity(
    user_id: str, redis: Redis, limit: int = 20
) -> list[ActivityLog]:
    """Get user's recent activities."""
    try:
        raw_activities = await get_list_items(
            redis,
            "activity",
            user_id,
            start=0,
            end=limit - 1,
        )
        return [ActivityLog(**item) for item in raw_activities]
    except DatabaseException as e:
        logger.error(f"Failed to get activities for user {user_id}: {e.detail}")
        return []


# ────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────


router = APIRouter(prefix="/api/v1/users", tags=["Users"])


@router.get("/{user_id}")
async def get_user_endpoint(
    user_id: str, redis: Redis = Depends(get_redis)
) -> UserProfile:
    """Get user profile with Redis caching.

    Endpoint example:
    GET /api/v1/users/123

    Behavior:
    - First request: fetches from DB, caches for 1 hour
    - Subsequent requests: served from cache (< 1ms vs ~100ms DB)
    - After 1 hour: refreshes from DB
    """
    try:
        user = await get_cached_user(user_id, redis)
        return user
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user")


@router.patch("/{user_id}")
async def update_user_endpoint(
    user_id: str,
    updates: UserUpdate,
    redis: Redis = Depends(get_redis),
) -> UserProfile:
    """Update user profile and refresh cache.

    Endpoint example:
    PATCH /api/v1/users/123
    {
        "name": "Jane Doe",
        "email": "jane@example.com"
    }

    Behavior:
    - Updates user in database
    - Refreshes cache immediately
    - Returns updated user profile
    """
    try:
        user = await update_cached_user(user_id, updates, redis)
        return user
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.delete("/{user_id}")
async def delete_user_endpoint(user_id: str, redis: Redis = Depends(get_redis)) -> dict:
    """Delete user and invalidate cache.

    Endpoint example:
    DELETE /api/v1/users/123

    Behavior:
    - Deletes user from database
    - Invalidates cached profile
    - Clears activity history cache
    """
    try:
        # Delete from DB (simulated)
        logger.info(f"Deleting user: {user_id}")

        # Invalidate all related caches
        await invalidate_user_cache(user_id, redis)
        await delete_list(redis, "activity", user_id)

        return {"status": "deleted", "user_id": user_id}
    except DatabaseException as e:
        logger.error(f"Failed to delete user {user_id}: {e.detail}")
        raise HTTPException(status_code=500, detail="Failed to delete user")


@router.post("/{user_id}/activity")
async def log_activity_endpoint(
    user_id: str,
    activity: ActivityLog,
    redis: Redis = Depends(get_redis),
) -> dict:
    """Log user activity.

    Endpoint example:
    POST /api/v1/users/123/activity
    {
        "timestamp": "2024-01-15T10:30:00Z",
        "action": "login",
        "resource": "auth",
        "details": {"ip": "192.168.1.1"}
    }

    Behavior:
    - Stores activity in Redis list (prepend for latest first)
    - Keeps 30 days of history
    - Returns confirmation
    """
    try:
        await add_activity(user_id, activity, redis)
        return {
            "status": "logged",
            "user_id": user_id,
            "action": activity.action,
        }
    except DatabaseException as e:
        logger.error(f"Failed to log activity: {e.detail}")
        raise HTTPException(status_code=500, detail="Failed to log activity")


@router.get("/{user_id}/activity")
async def get_activity_endpoint(
    user_id: str, limit: int = 20, redis: Redis = Depends(get_redis)
) -> dict:
    """Get user's recent activity.

    Endpoint example:
    GET /api/v1/users/123/activity?limit=10

    Behavior:
    - Returns user's recent activities from cache
    - Limits to N most recent items (default: 20)
    - Returns empty list if none found
    """
    try:
        activities = await get_user_activity(user_id, redis, limit)
        return {
            "user_id": user_id,
            "total": len(activities),
            "activities": activities,
        }
    except Exception as e:
        logger.error(f"Failed to get activities for user {user_id}: {e!s}")
        return {"user_id": user_id, "total": 0, "activities": []}


# ────────────────────────────────────────────────────────────
# Batch operations example
# ────────────────────────────────────────────────────────────


@router.get("/batch/users")
async def get_multiple_users(
    user_ids: list[str], redis: Redis = Depends(get_redis)
) -> dict:
    """Get multiple users efficiently.

    Query example:
    GET /api/v1/users/batch/users?user_ids=1&user_ids=2&user_ids=3

    Behavior:
    - Fetches all users (from cache or DB)
    - Returns list of users found
    """
    users = {}
    failed = []

    for user_id in user_ids:
        try:
            user = await get_cached_user(user_id, redis)
            users[user_id] = user.model_dump()
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e!s}")
            failed.append(user_id)

    return {
        "total": len(users),
        "users": users,
        "failed": failed,
    }


# ────────────────────────────────────────────────────────────
# Advanced: Cache warming (pre-populate cache on startup)
# ────────────────────────────────────────────────────────────


async def warm_user_cache(user_ids: list[str], redis: Redis) -> None:
    """Pre-populate cache with frequently accessed users.

    Call this during app startup or scheduled task.
    """
    logger.info(f"Warming cache for {len(user_ids)} users")

    for user_id in user_ids:
        try:
            user = await get_user_from_db(user_id)
            await set_cache(
                redis,
                "user",
                user_id,
                user.model_dump(),
                expire_seconds=3600,
            )
        except Exception as e:
            logger.error(f"Failed to warm cache for user {user_id}: {e!s}")


# ────────────────────────────────────────────────────────────
# Advanced: Cache invalidation strategies
# ────────────────────────────────────────────────────────────


async def invalidate_all_user_caches(redis: Redis) -> None:
    """Mass invalidate all user caches (use carefully!)."""
    # In production, use Redis key patterns:
    # keys = await redis.keys("user:*")
    # await redis.delete(*keys)
    logger.warning("Invalidating all user caches")
    # For now, just log


async def cache_with_fallback(
    user_id: str, redis: Redis, timeout: int = 3600
) -> UserProfile | None:
    """Get user with automatic cache fallback pattern.

    This pattern:
    1. Tries cache
    2. If miss, tries DB
    3. If DB fails too, returns None
    4. Logs all failures for monitoring
    """
    try:
        return await get_cached_user(user_id, redis)
    except Exception as e:
        logger.error(
            f"Complete failure retrieving user {user_id}: {e!s}",
            user_id=user_id,
        )
        return None
