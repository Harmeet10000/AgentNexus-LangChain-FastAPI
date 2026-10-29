"""Shared rate limiting.

Two semantics, one package — they are not interchangeable:

- ``RateLimitService`` + ``get_rate_limiter()``: GCRA leaky-bucket HTTP throttle,
  keyed ``rl:gcra:*``, fail-open, used as a FastAPI dependency.
- ``RateLimiter`` + ``build_rate_limiter()``: fixed-window provider quota,
  keys ``rate:*:min/hour``, used by the crawler and search paths.

Layout: ``application/`` holds the policy (scopes, window configuration);
``adapters/redis/`` holds the Redis-bound limiters and the FastAPI dependency.
"""

from app.shared.rate_limit.adapters.redis.dependencies import get_rate_limiter
from app.shared.rate_limit.adapters.redis.quota import RateLimiter, build_rate_limiter
from app.shared.rate_limit.adapters.redis.throttle import GCRA_LUA_SCRIPT, RateLimitService
from app.shared.rate_limit.application.policy import (
    RATE_LIMIT_CONFIGS,
    RateLimitConfig,
    RateLimitScope,
)

__all__ = [
    "GCRA_LUA_SCRIPT",
    "RATE_LIMIT_CONFIGS",
    "RateLimitConfig",
    "RateLimitScope",
    "RateLimitService",
    "RateLimiter",
    "build_rate_limiter",
    "get_rate_limiter",
]
