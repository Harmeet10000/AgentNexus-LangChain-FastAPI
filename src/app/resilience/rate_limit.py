"""Resilience: rate limiting (throttle + quota), in one module.

Two semantics, one home — they are not interchangeable:
- `RateLimitService` + `get_rate_limiter()`: GCRA leaky-bucket HTTP throttle,
  keyed `rl:gcra:*`, fail-open, used as a FastAPI dependency.
- `RateLimiter` + `build_rate_limiter()`: fixed-window provider quota,
  keys `rate:*:min/hour`, used by the crawler and search paths.
"""

import math
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ValidationError
from redis import RedisError
from redis.asyncio import Redis

from app.config import get_settings
from app.security import tokens
from app.utils import logger
from app.utils.exceptions import TooManyRequestsException, UnauthorizedException

# ── Quota (fixed-window provider limits) ──

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


class RateLimiter:
    """Rate limiter using fastapi-limiter and Redis backend."""

    def __init__(self, redis_client: Redis | None = None):
        self.redis_client = redis_client

    async def check_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
    ) -> tuple[bool, dict[str, Any]]:
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

    async def acquire(
        self,
        identifier: str,
        scope: RateLimitScope,
    ) -> tuple[bool, dict[str, Any]]:
        """Atomically consume one request from both rate-limit windows."""
        if not self.redis_client:
            return True, {}

        config = RATE_LIMIT_CONFIGS.get(scope)
        if config is None:
            return True, {}

        minute_key = f"rate:{scope.value}:{identifier}:min"
        hour_key = f"rate:{scope.value}:{identifier}:hour"
        # NOTE: expiry is set only on the first increment of a window. Setting
        # EXPIRE on every call would extend the window on each request and pin
        # steady traffic above the limit forever.
        pipe = self.redis_client.pipeline(transaction=True)
        pipe.incr(minute_key)
        pipe.incr(hour_key)
        minute_count, hour_count = await pipe.execute()
        if minute_count == 1:
            await self.redis_client.expire(minute_key, 60)
        if hour_count == 1:
            await self.redis_client.expire(hour_key, 3600)

        if minute_count > config.per_minute or hour_count > config.per_hour:
            exceeded_minute = minute_count > config.per_minute
            retry_after = await self.redis_client.ttl(minute_key if exceeded_minute else hour_key)
            return False, {
                "error": "Rate limit exceeded",
                "limit": config.per_minute if exceeded_minute else config.per_hour,
                "window": "minute" if exceeded_minute else "hour",
                "retry_after": max(int(retry_after), 0),
            }

        return True, {
            "remaining_minute": config.per_minute - minute_count,
            "remaining_hour": config.per_hour - hour_count,
        }

    async def increment_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
    ) -> None:
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
    ) -> dict[str, int]:
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


def build_rate_limiter(redis_client: Redis | None = None) -> RateLimiter:
    """Get rate limiter instance."""
    global _rate_limiter  # noqa: PLW0603
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(redis_client)
    elif redis_client is not None:
        _rate_limiter.redis_client = redis_client
    return _rate_limiter

# ── Throttle (GCRA leaky bucket) ──

# GCRA Leaky Bucket Algorithm (O(1) Memory, Atomic)
# Returns: { 1: allowed, 0: rejected }, { retry_after_ms }
GCRA_LUA_SCRIPT = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local rate = tonumber(ARGV[3])
local period_ms = tonumber(ARGV[4])

local emission_interval_ms = period_ms / rate
local burst_tolerance_ms = emission_interval_ms * burst

local tat = redis.call("GET", key)
if not tat then
    tat = now_ms
else
    tat = tonumber(tat)
end

local new_tat = math.max(tat, now_ms) + emission_interval_ms
local allow_at = new_tat - burst_tolerance_ms

if now_ms < allow_at then
    local retry_after_ms = math.ceil(allow_at - now_ms)
    return {0, retry_after_ms}
end

-- Set TTL to clean up the key once the bucket is fully empty
local ttl_ms = math.ceil(new_tat - now_ms)
redis.call("SET", key, new_tat, "PX", ttl_ms)

return {1, 0}
"""


def _raise_rate_limit_exceeded(retry_seconds: int) -> None:
    raise TooManyRequestsException(
        detail="Rate limit exceeded. Please slow down.",
        headers={"Retry-After": str(retry_seconds)},
    )


class RateLimitService:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis
        # Pre-load the script into Redis memory for fast SHA execution
        self._script = self.redis.register_script(GCRA_LUA_SCRIPT)

    async def check_limit(
        self, identifier: str, burst: int, rate: int, period_seconds: int
    ) -> None:
        """
        Evaluates the rate limit using GCRA. Raises TooManyRequestsException if exceeded.
        """
        now_ms = int(time.time() * 1000)
        period_ms = period_seconds * 1000
        redis_key = f"rl:gcra:{identifier}"

        try:
            allowed, retry_after_ms = await self._script(
                keys=[redis_key], args=[now_ms, burst, rate, period_ms]
            )

            if not allowed:
                retry_seconds = math.ceil(retry_after_ms / 1000)
                logger.bind(identifier=identifier, retry_in=retry_seconds).warning(
                    "Rate limit exceeded"
                )
                _raise_rate_limit_exceeded(retry_seconds)

        except TooManyRequestsException:
            raise
        except (RedisError, ConnectionError, TimeoutError) as e:
            # Fail open to prevent a Redis outage from taking down the API
            logger.bind(error=str(e)).exception("Redis rate limiting failed, bypassing")
_ACCESS_TOKEN_TYPE = "access"  # noqa: S105 — token type constant, not a password


def _extract_presented_token(request: Request) -> str | None:
    """Bearer header first, access-token cookie second; None when neither is present."""
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
    cookie_token: str | None = request.cookies.get("access_token")
    return cookie_token or None


def _resolve_user_identifier(request: Request) -> str | None:
    """Subject of a valid access token, namespaced, or None when the request carries none.

    Identity comes from validated token claims, never from per-request state.
    Fail-open: an absent, expired, or malformed token — or a token of the wrong
    type — yields None so the caller falls back to IP keying instead of
    rejecting the request. A rate limiter must not become an auth gate,
    especially on the unauthenticated endpoints that use it.
    """
    raw_token = _extract_presented_token(request)
    if raw_token is None:
        return None
    try:
        claims = tokens.decode_token(raw_token)
    except (UnauthorizedException, ValidationError, KeyError):
        return None
    if claims.token_type != _ACCESS_TOKEN_TYPE:
        return None
    return f"user:{claims.sub}"


def get_rate_limiter(
    burst: int = 10,
    rate: int = 5,
    period: int = 60,
    *,
    scope: str | None = None,
) -> Callable[[Request], Awaitable[None]]:
    """
    Build a FastAPI dependency that enforces a Redis-backed rate limit.

    Identity is per-user when the request carries a valid access token
    (keyed ``user:<subject>``) and per-IP otherwise, so the limiter works
    unchanged on unauthenticated endpoints.

    Args:
        burst: Maximum burst capacity before requests are throttled.
        rate: Number of tokens refilled per period.
        period: Refill period in seconds.
        scope: Optional independent quota namespace shared by endpoints using it.
    """

    async def rate_limit_dependency(request: Request) -> None:
        redis_client = request.app.state.redis
        forwarded_for = request.headers.get("X-Forwarded-For")
        client_ip = request.client.host if request.client else "unknown"
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip() or client_ip

        identifier = _resolve_user_identifier(request) or client_ip
        if scope is not None:
            identifier = f"{scope}:{identifier}"
        service = RateLimitService(redis_client)
        await service.check_limit(
            identifier=identifier, burst=burst, rate=rate, period_seconds=period
        )

    return rate_limit_dependency
