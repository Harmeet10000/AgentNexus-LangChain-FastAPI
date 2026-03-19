from collections.abc import Awaitable, Callable

from fastapi import Request

from .service import RateLimitService


def get_rate_limiter(
    burst: int = 10, rate: int = 5, period: int = 60
) -> Callable[[Request], Awaitable[None]]:
    """
    Build a FastAPI dependency that enforces a Redis-backed rate limit.

    Args:
        burst: Maximum burst capacity before requests are throttled.
        rate: Number of tokens refilled per period.
        period: Refill period in seconds.
    """

    async def rate_limit_dependency(request: Request) -> None:
        redis_client = request.app.state.redis
        forwarded_for = request.headers.get("X-Forwarded-For")
        client_ip = request.client.host if request.client else "unknown"
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip() or client_ip

        identifier = getattr(request.state, "user_id", None) or client_ip
        service = RateLimitService(redis_client)
        await service.check_limit(
            identifier=identifier, burst=burst, rate=rate, period_seconds=period
        )

    return rate_limit_dependency
