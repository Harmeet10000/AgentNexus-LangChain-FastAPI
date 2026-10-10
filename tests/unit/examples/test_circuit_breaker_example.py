import asyncio

import pytest
from fakeredis.aioredis import FakeRedis

from app.examples.circuit_breaker import (
    CircuitBreakerOpenError,
    CircuitBreakerService,
    CircuitBreakerSettings,
    RedisCircuitBreakerStore,
)


@pytest.mark.anyio
async def test_breaker_opens_recovers_and_allows_only_one_probe() -> None:
    redis_client = FakeRedis()
    breaker = CircuitBreakerService(
        RedisCircuitBreakerStore(redis_client),
        CircuitBreakerSettings(
            failure_threshold=2,
            recovery_timeout_seconds=1,
            probe_lease_seconds=5,
        ),
    )

    async def fail() -> None:
        msg = "downstream unavailable"
        raise OSError(msg)

    with pytest.raises(OSError):
        await breaker.call("payments", fail)
    with pytest.raises(OSError):
        await breaker.call("payments", fail)
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call("payments", fail)

    await redis_client.set("examples:circuit:{payments}:open-until", "0")
    probe_entered = asyncio.Event()
    finish_probe = asyncio.Event()

    async def successful_probe() -> str:
        probe_entered.set()
        await finish_probe.wait()
        return "recovered"

    probe = asyncio.create_task(breaker.call("payments", successful_probe))
    await probe_entered.wait()
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call("payments", successful_probe)

    finish_probe.set()
    assert await probe == "recovered"
    assert await breaker.call("payments", lambda: _return("closed")) == "closed"
    await redis_client.aclose()


@pytest.mark.anyio
async def test_breaker_can_use_a_redis_client_with_default_byte_responses() -> None:
    redis_client = FakeRedis()
    breaker = CircuitBreakerService(RedisCircuitBreakerStore(redis_client))

    assert await breaker.call("search", lambda: _return("ok")) == "ok"
    await redis_client.aclose()


async def _return(value: str) -> str:
    return value


def test_settings_reject_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreakerSettings(failure_threshold=0)
