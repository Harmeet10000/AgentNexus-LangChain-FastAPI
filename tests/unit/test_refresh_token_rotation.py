"""Deterministic tests for refresh-token family rotation and reuse handling."""

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
from app.features.auth.adapters.security.tokens import derive_refresh_token_material

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


async def _seed_family(
    repo: RefreshTokenRepository,
    redis: FakeRedis,
    *,
    family_id: str,
) -> tuple[SessionData, str, str, str, str]:
    """Store a session plus its generation-0 refresh family; return rotation material."""
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
    return session, token_id, token_secret, child_id, child_secret


async def _family_status(redis: FakeRedis, family_id: str) -> str:
    raw = await redis.hget(f"auth:refresh_family:{family_id}", "status")
    if isinstance(raw, bytes):
        return raw.decode()
    assert isinstance(raw, str)
    return raw


@pytest.mark.asyncio
async def test_rotation_stores_only_digests_and_is_atomic(redis: FakeRedis) -> None:
    repo = RefreshTokenRepository(redis)
    family_id = "family-1"
    session, token_id, token_secret, child_id, child_secret = await _seed_family(
        repo, redis, family_id=family_id
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
async def test_concurrent_refresh_reuse_revokes_family_and_session(redis: FakeRedis) -> None:
    """A superseded presentation racing the winner is reuse, never a grace retry."""
    repo = RefreshTokenRepository(redis)
    family_id = "family-concurrent"
    session, token_id, token_secret, child_id, child_secret = await _seed_family(
        repo, redis, family_id=family_id
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
            )
            for _ in range(2)
        )
    )
    outcomes = [result.unwrap() for result in results]
    assert sorted(outcome.status for outcome in outcomes) == ["reused", "rotated"]
    assert {outcome.generation for outcome in outcomes if outcome.status == "rotated"} == {1}
    assert await redis.exists(f"auth:session:{session.session_id}") == 0
    assert await _family_status(redis, family_id) == "revoked"


@pytest.mark.asyncio
async def test_immediate_replay_of_superseded_token_revokes_family(redis: FakeRedis) -> None:
    """Re-presenting the just-rotated token — even within the same second — is reuse."""
    repo = RefreshTokenRepository(redis)
    family_id = "family-immediate-replay"
    session, token_id, token_secret, child_id, child_secret = await _seed_family(
        repo, redis, family_id=family_id
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
    )
    assert isinstance(rotated, Success)
    assert rotated.unwrap().status == "rotated"

    replay = await repo.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        expected_authz_revision=4,
        now=session.created_at,
    )
    assert isinstance(replay, Success)
    assert replay.unwrap().status == "reused"
    assert await redis.exists(f"auth:session:{session.session_id}") == 0
    assert await _family_status(redis, family_id) == "revoked"


@pytest.mark.asyncio
async def test_older_generation_replay_revokes_after_family_advances(redis: FakeRedis) -> None:
    repo = RefreshTokenRepository(redis)
    family_id = "family-older-replay"
    session, token_id, token_secret, child_id, child_secret = await _seed_family(
        repo, redis, family_id=family_id
    )
    grandchild_id, grandchild_secret = derive_refresh_token_material(family_id, 2)
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
    )
    assert isinstance(replay, Success)
    assert replay.unwrap().status == "reused"
