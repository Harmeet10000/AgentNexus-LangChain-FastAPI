"""Deterministic tests for refresh-token family rotation and replay handling."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from returns.result import Success

from app.features.auth.adapters.persistence.mongodb.account_repository import (
    RefreshTokenRepository,
    SessionData,
)
from app.security.tokens import derive_refresh_token_material

if TYPE_CHECKING:
    from fakeredis.aioredis import FakeRedis


def _session(*, family_id: str, user_id: str = "507f1f77bcf86cd799439011") -> SessionData:
    now = datetime.now(UTC)
    return SessionData(
        session_id="session-1",
        user_id=user_id,
        device_id="device-1",
        created_at=now,
        expires_at=now + timedelta(hours=1),
        ttl=3600,
        authz_revision=4,
        family_id=family_id,
    )


@pytest.mark.asyncio
async def test_rotation_stores_only_digests_and_is_atomic(redis: FakeRedis) -> None:
    repo = RefreshTokenRepository(redis)
    family_id = "family-1"
    token_id, token_secret = derive_refresh_token_material(family_id, 0)
    child_id, child_secret = derive_refresh_token_material(family_id, 1)
    session = _session(family_id=family_id)
    assert isinstance(await repo.store_session(session), Success)
    assert isinstance(
        await repo.create_refresh_family(
            family_id=family_id,
            user_id=session.user_id,
            session_id=session.session_id,
            device_id=session.device_id,
            authz_revision=session.authz_revision,
            token_id=token_id,
            token_secret=token_secret,
            issued_at=session.created_at,
            expires_at=session.expires_at,
        ),
        Success,
    )

    first = await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        expected_authz_revision=4,
        now=session.created_at,
        grace_seconds=10,
    )
    assert isinstance(first, Success)
    assert first.unwrap().status == "rotated"
    stored = await redis.hgetall(f"auth:refresh_family:{family_id}")
    values = {
        (key.decode() if isinstance(key, bytes) else key): (
            value.decode() if isinstance(value, bytes) else value
        )
        for key, value in stored.items()
    }
    assert token_id not in values.values()
    assert token_secret not in values.values()
    assert values["current_id_hash"] != child_id


@pytest.mark.asyncio
async def test_concurrent_refresh_has_one_rotation_and_one_idempotent_grace(
    redis: FakeRedis,
) -> None:
    repo = RefreshTokenRepository(redis)
    family_id = "family-concurrent"
    token_id, token_secret = derive_refresh_token_material(family_id, 0)
    child_id, child_secret = derive_refresh_token_material(family_id, 1)
    session = _session(family_id=family_id)
    await repo.store_session(session)
    await repo.create_refresh_family(
        family_id=family_id,
        user_id=session.user_id,
        session_id=session.session_id,
        device_id=session.device_id,
        authz_revision=session.authz_revision,
        token_id=token_id,
        token_secret=token_secret,
        issued_at=session.created_at,
        expires_at=session.expires_at,
    )

    results = await asyncio.gather(
        *(
            repo.rotate_refresh_token(
                family_id=family_id,
                token_id=token_id,
                token_secret=token_secret,
                child_token_id=child_id,
                child_token_secret=child_secret,
                child_generation=1,
                expected_authz_revision=4,
                now=session.created_at,
                grace_seconds=10,
            )
            for _ in range(2)
        )
    )
    outcomes = [result.unwrap() for result in results]
    assert sorted(outcome.status for outcome in outcomes) == ["grace", "rotated"]
    assert {outcome.generation for outcome in outcomes} == {1}


@pytest.mark.asyncio
async def test_zero_grace_replay_is_not_accepted_within_same_second(
    redis: FakeRedis,
) -> None:
    repo = RefreshTokenRepository(redis)
    family_id = "family-zero-grace"
    token_id, token_secret = derive_refresh_token_material(family_id, 0)
    child_id, child_secret = derive_refresh_token_material(family_id, 1)
    session = _session(family_id=family_id)
    await repo.store_session(session)
    await repo.create_refresh_family(
        family_id=family_id,
        user_id=session.user_id,
        session_id=session.session_id,
        device_id=session.device_id,
        authz_revision=session.authz_revision,
        token_id=token_id,
        token_secret=token_secret,
        issued_at=session.created_at,
        expires_at=session.expires_at,
    )
    rotated = await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        expected_authz_revision=4,
        now=session.created_at,
        grace_seconds=0,
    )
    assert isinstance(rotated, Success)
    replay = await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        expected_authz_revision=4,
        now=session.created_at,
        grace_seconds=0,
    )
    assert isinstance(replay, Success)
    assert replay.unwrap().status == "reused"


@pytest.mark.asyncio
async def test_replay_after_grace_revokes_family_and_session(redis: FakeRedis) -> None:
    repo = RefreshTokenRepository(redis)
    family_id = "family-replay"
    token_id, token_secret = derive_refresh_token_material(family_id, 0)
    child_id, child_secret = derive_refresh_token_material(family_id, 1)
    session = _session(family_id=family_id)
    await repo.store_session(session)
    await repo.create_refresh_family(
        family_id=family_id,
        user_id=session.user_id,
        session_id=session.session_id,
        device_id=session.device_id,
        authz_revision=session.authz_revision,
        token_id=token_id,
        token_secret=token_secret,
        issued_at=session.created_at,
        expires_at=session.expires_at,
    )
    await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        expected_authz_revision=4,
        now=session.created_at,
        grace_seconds=1,
    )
    replay = await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        expected_authz_revision=4,
        now=session.created_at + timedelta(seconds=2),
        grace_seconds=1,
    )
    assert isinstance(replay, Success)
    assert replay.unwrap().status == "reused"
    assert await redis.exists(f"auth:session:{session.session_id}") == 0
    assert (await redis.hget(f"auth:refresh_family:{family_id}", "status")) == "revoked"


@pytest.mark.asyncio
async def test_older_generation_replay_revokes_after_family_advances(redis: FakeRedis) -> None:
    repo = RefreshTokenRepository(redis)
    family_id = "family-older-replay"
    session = _session(family_id=family_id)
    token_id, token_secret = derive_refresh_token_material(family_id, 0)
    child_id, child_secret = derive_refresh_token_material(family_id, 1)
    grandchild_id, grandchild_secret = derive_refresh_token_material(family_id, 2)
    await repo.store_session(session)
    await repo.create_refresh_family(
        family_id=family_id,
        user_id=session.user_id,
        session_id=session.session_id,
        device_id=session.device_id,
        authz_revision=4,
        token_id=token_id,
        token_secret=token_secret,
        issued_at=session.created_at,
        expires_at=session.expires_at,
    )
    await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        presented_generation=0,
        expected_authz_revision=4,
        now=session.created_at,
        grace_seconds=1,
    )
    await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=child_id,
        token_secret=child_secret,
        child_token_id=grandchild_id,
        child_token_secret=grandchild_secret,
        child_generation=2,
        presented_generation=1,
        expected_authz_revision=4,
        now=session.created_at,
        grace_seconds=1,
    )
    replay = await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        presented_generation=0,
        expected_authz_revision=4,
        now=session.created_at + timedelta(seconds=2),
        grace_seconds=1,
    )
    assert isinstance(replay, Success)
    assert replay.unwrap().status == "reused"
