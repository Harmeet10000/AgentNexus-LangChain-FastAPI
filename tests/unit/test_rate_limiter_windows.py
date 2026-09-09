"""Rate-limit windows expire instead of extending forever."""

from __future__ import annotations

from fakeredis.aioredis import FakeRedis

from app.shared.services.rate_limiter import RateLimiter, RateLimitScope


class _SpyRedis(FakeRedis):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.expire_calls: list[str] = []

    async def expire(self, name: str, *args: object, **kwargs: object) -> bool:  # type: ignore[override]
        self.expire_calls.append(name)
        return await super().expire(name, *args, **kwargs)  # type: ignore[arg-type]


async def test_window_expiry_is_set_once_not_extended_per_request() -> None:
    redis = _SpyRedis(decode_responses=True)
    limiter = RateLimiter(redis_client=redis)

    allowed, _ = await limiter.acquire("steady-client", RateLimitScope.CRAWL)
    assert allowed is True

    # Steady traffic must not push the window out: EXPIRE runs once, on the
    # first increment. (On the old code this records 6 calls per key and the
    # key never expires while traffic flows.)
    for _ in range(5):
        await limiter.acquire("steady-client", RateLimitScope.CRAWL)
    assert redis.expire_calls.count("rate:crawl:steady-client:min") == 1
    assert redis.expire_calls.count("rate:crawl:steady-client:hour") == 1


async def test_exceeded_limit_names_the_right_window_and_delay() -> None:
    redis = FakeRedis(decode_responses=True)
    limiter = RateLimiter(redis_client=redis)

    info: dict[str, object] = {}
    for _ in range(200):
        allowed, info = await limiter.acquire("bursty-client", RateLimitScope.CRAWL)
        if not allowed:
            break
    assert allowed is False
    assert info["window"] == "minute"
    assert 0 <= int(info["retry_after"]) <= 60
