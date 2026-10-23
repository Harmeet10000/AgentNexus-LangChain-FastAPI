"""ThreadLock unit tests (doc §5.5 exclusion fence, fail-closed aliveness).

FakeRedis supplies real SET NX/EX semantics; the renewal loop is exercised at
the `_renew_once` level because wall-clock sleeps have no place in unit tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis
from redis.exceptions import ConnectionError as RedisConnectionError

from app.features.agent_saul.thread_lock import DEFAULT_TTL_SECONDS, ThreadLock, lock_key

pytestmark = pytest.mark.unit


async def test_renew_survives_a_bytes_returning_client() -> None:
    """Regression (§5.5): the production pool (redis_fastapi) is built WITHOUT
    decode_responses, so the same lock key reads back as bytes here and as str
    through every other client. The ownership compare must treat b'…' and
    '…' as the same token, or every first renewal at TTL/3 declared the lease
    lost mid-stream and the run died with LOCK_LOST ~20s in.
    """
    for decode_responses in (True, False):
        client = FakeRedis(decode_responses=decode_responses)
        lock = ThreadLock(client, "t-bytes")
        try:
            assert await lock.acquire() is True
            assert await lock._renew_once() is True
            assert lock.is_held is True
        finally:
            await lock.release()
            await client.aclose()


async def make_lock(test_redis, thread_id: str = "t-1", **kwargs) -> ThreadLock:
    return ThreadLock(test_redis, thread_id, **kwargs)


async def test_acquire_sets_key_and_marks_fresh(redis) -> None:
    lock = await make_lock(redis)
    try:
        assert await lock.acquire() is True
        assert lock.is_held is True
        assert lock.heartbeat.fresh is True
        assert lock.is_alive() is True
        assert await redis.exists(lock_key("t-1")) == 1
    finally:
        await lock.release()


async def test_second_connection_refused_while_held(redis) -> None:
    first = await make_lock(redis)
    second = await make_lock(redis, thread_id="t-2")
    try:
        assert await first.acquire() is True
        # Same thread name → the exclusion fence bites.
        rival = ThreadLock(redis, "t-1")
        try:
            assert await rival.acquire() is False
            # A refused rival's release must never delete the holder's key
            # (§5.5: ownership token guards DEL; TTL is the correctness fence).
            await rival.release()
            assert await redis.exists(lock_key("t-1")) == 1  # holder's key alive
            assert first.is_held is True  # holder unaffected
        finally:
            await rival.release()
        # A different thread still acquires independently.
        assert await second.acquire() is True
    finally:
        await first.release()
        await second.release()


async def test_release_frees_the_fence(redis) -> None:
    lock = await make_lock(redis)
    await lock.acquire()
    await lock.release()
    assert lock.is_held is False
    assert lock.is_alive() is False
    assert await redis.exists(lock_key("t-1")) == 0
    # A following connection wins the fence.
    follower = await make_lock(redis)
    try:
        assert await follower.acquire() is True
    finally:
        await follower.release()


async def test_release_is_best_effort_on_outage(redis) -> None:
    lock = await make_lock(redis)
    await lock.acquire()
    redis.delete = AsyncMock(side_effect=RedisConnectionError("down"))
    # Must not raise; the TTL is the real fence anyway.
    await lock.release()
    assert lock.is_held is False


async def test_acquire_fails_closed_on_redis_outage(redis) -> None:
    lock = await make_lock(redis)
    redis.set = AsyncMock(side_effect=RedisConnectionError("down"))
    assert await lock.acquire() is False
    assert lock.is_held is False
    assert lock.heartbeat.fresh is False


async def test_renew_once_extends_only_when_key_present(redis) -> None:
    lock = await make_lock(redis)
    await lock.acquire()
    try:
        assert await lock._renew_once() is True
        # FakeRedis TTL: a fresh EX leaves a positive countdown.
        assert await redis.ttl(lock_key("t-1")) > 0
    finally:
        await lock.release()


async def test_renew_once_reports_loss_when_key_gone(redis) -> None:
    lock = await make_lock(redis)
    await lock.acquire()
    try:
        await redis.delete(lock_key("t-1"))  # simulated crash of the key holder
        assert await lock._renew_once() is False
    finally:
        await lock.release()


async def test_ttl_default_is_doc_value() -> None:
    assert DEFAULT_TTL_SECONDS == 60


async def test_custom_ttl_derives_renewal_interval(redis) -> None:
    lock = ThreadLock(redis, "t-custom-ttl", ttl_seconds=9)
    assert lock._renew_interval == 3


async def test_explicit_renewal_interval_overrides_ttl_derivation(redis) -> None:
    lock = ThreadLock(redis, "t-custom-renew", ttl_seconds=9, renew_interval=2.5)
    assert lock._renew_interval == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ttl_seconds": 0}, "ttl_seconds must be positive"),
        ({"renew_interval": 0}, "renew_interval must be positive"),
    ],
)
async def test_lock_rejects_non_positive_timing(redis, kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ThreadLock(redis, "t-invalid-timing", **kwargs)
