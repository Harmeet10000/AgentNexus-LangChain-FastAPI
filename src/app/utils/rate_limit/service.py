import math
import time

from loguru import logger
from redis.asyncio import Redis

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
        except Exception as e:
            # Fail open to prevent a Redis outage from taking down the API
            logger.bind(error=str(e)).error("Redis rate limiting failed, bypassing.")
