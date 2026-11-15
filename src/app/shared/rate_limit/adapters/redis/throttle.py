"""GCRA leaky-bucket HTTP throttle, Redis-backed (keys ``rl:gcra:*``, fail-open)."""

import math
import time

from redis import RedisError
from redis.asyncio import Redis

from app.observability.logging import logger
from app.utils.exceptions import TooManyRequestsException

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
            # Deliberate policy decision (G8, 2026-11-15 smoke): rate limiting
            # fails OPEN during a Redis outage — availability over
            # enforcement.  The observed cost is that quota enforcement is
            # silently disabled for the outage's duration; the alternative
            # (fail closed) turns every limiter-guarded route into a 503 and
            # magnifies an infrastructure failure into a full API outage.
            # Contrast with AUTH session storage (app/connections/redis.py),
            # which fails closed as honest typed 503 degradation — sessions
            # cannot be served without Redis, and faking availability would
            # break the auth invariant rather than preserve it.
            logger.bind(error=str(e)).exception("Redis rate limiting failed, bypassing")
