"""Deterministic fencing and retry tests for pending session revocations."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fakeredis.aioredis import FakeRedis
from returns.result import Failure, Success

from app.features.auth.application.revocation.pending import (
    PendingRevocationConfig,
    PendingRevocationReconciler,
)
from app.features.auth.domain.errors import AuthInfrastructureError

pytestmark = pytest.mark.unit


def _user(revision: int = 3, *, pending: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        id="507f1f77bcf86cd799439011",
        role_revision=revision,
        session_revocation_pending=pending,
        updated_at=datetime.now(UTC) - timedelta(seconds=30),
    )


def _reconciler(
    redis: FakeRedis,
    user: SimpleNamespace,
    *,
    revoke,
    clear,
    now: list[float] | None = None,
    **config,
) -> PendingRevocationReconciler:
    clock = now or [1_000.0]
    return PendingRevocationReconciler(
        redis,
        list_pending=lambda _limit, _offset: _async_return([user]),
        get_user=lambda _user_id: _async_return(user),
        revoke_sessions=revoke,
        clear_marker=clear,
        config=PendingRevocationConfig(
            backoff_base_seconds=10,
            backoff_max_seconds=60,
            jitter_seconds=0,
            **config,
        ),
        clock=lambda: clock[0],
        monotonic=lambda: 0.0,
        random=lambda _upper: 0,
    )


async def _async_return(value):
    return value


@pytest.mark.asyncio
async def test_recovery_revokes_and_clears_exact_revision() -> None:
    redis = FakeRedis(decode_responses=True)
    user = _user()
    calls: list[tuple[str, int]] = []

    async def revoke(user_id: str, revision: int):
        calls.append((user_id, revision))
        return Success(user)

    async def clear(current, revision: int):
        assert current.role_revision == revision == 3
        current.session_revocation_pending = False
        return Success(current)

    report = await _reconciler(redis, user, revoke=revoke, clear=clear).run()

    assert report["succeeded"] == 1
    assert calls == [(str(user.id), 3)]
    assert user.session_revocation_pending is False
    assert await redis.get("auth:session_revocation:retry:" + str(user.id)) is None
    await redis.aclose()


@pytest.mark.asyncio
async def test_redis_outage_retries_with_backoff_and_force_bypasses_delay() -> None:
    redis = FakeRedis(decode_responses=True)
    user = _user()
    clock = [1_000.0]
    attempts = 0

    async def revoke(_user_id: str, _revision: int):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return Failure(AuthInfrastructureError(message="redis unavailable"))
        return Success(user)

    async def clear(current, _revision: int):
        current.session_revocation_pending = False
        return Success(current)

    reconciler = _reconciler(
        redis,
        user,
        revoke=revoke,
        clear=clear,
        now=clock,
    )
    first = await reconciler.run()
    second = await reconciler.run()
    forced = await reconciler.run(force=True)

    assert first["retryable_failure"] == 1
    assert second["skipped"] == 1
    assert forced["succeeded"] == 1
    assert attempts == 2
    await redis.aclose()


@pytest.mark.asyncio
async def test_newer_revision_never_gets_cleared_by_stale_worker() -> None:
    redis = FakeRedis(decode_responses=True)
    discovered = _user(3)
    current = _user(4)
    clear_calls = 0

    async def revoke(_user_id: str, _revision: int):
        pytest.fail("stale revision must be fenced before revoke")

    async def clear(_current, _revision: int):
        nonlocal clear_calls
        clear_calls += 1
        return Success(current)

    lookups = 0

    async def get_user(_user_id: str):
        nonlocal lookups
        lookups += 1
        return current

    reconciler = PendingRevocationReconciler(
        redis,
        list_pending=lambda _limit, _offset: _async_return([discovered]),
        get_user=get_user,
        revoke_sessions=revoke,
        clear_marker=clear,
        monotonic=lambda: 0.0,
    )

    report = await reconciler.run()

    assert report["stale_revision"] == 1
    assert clear_calls == 0
    assert current.session_revocation_pending is True
    await redis.aclose()


@pytest.mark.asyncio
async def test_duplicate_workers_are_bounded_by_per_user_lease() -> None:
    redis = FakeRedis(decode_responses=True)
    user = _user()
    entered = asyncio.Event()
    release = asyncio.Event()
    revokes = 0

    async def revoke(_user_id: str, _revision: int):
        nonlocal revokes
        revokes += 1
        entered.set()
        await release.wait()
        return Success(user)

    async def clear(current, _revision: int):
        current.session_revocation_pending = False
        return Success(current)

    first = _reconciler(redis, user, revoke=revoke, clear=clear)
    second = _reconciler(redis, user, revoke=revoke, clear=clear)
    first_task = asyncio.create_task(first.run())
    await entered.wait()
    second_report = await second.run()
    release.set()
    first_report = await first_task

    assert second_report["skipped"] == 1
    assert first_report["succeeded"] == 1
    assert revokes == 1
    await redis.aclose()


@pytest.mark.asyncio
async def test_cleared_users_do_not_leave_retry_records_forever() -> None:
    redis = FakeRedis(decode_responses=True)
    user = _user(pending=False)
    retry_key = "auth:session_revocation:retry:" + str(user.id)
    await redis.set(retry_key, '{"revision":3,"attempts":2}')

    reconciler = _reconciler(
        redis,
        user,
        revoke=lambda *_args: _async_return(Success(user)),
        clear=lambda *_args: _async_return(Success(user)),
    )
    report = await reconciler.run()

    assert report["cleaned_retry_records"] == 1
    assert await redis.get(retry_key) is None
    await redis.aclose()
