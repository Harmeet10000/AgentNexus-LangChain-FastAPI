"""Real Redis failure and recovery checks for the session store.

The suite is deliberately opt-in.  A local checkout can run it with
``LIVE_REDIS_URL=redis://127.0.0.1:6379/15``; CI supplies a disposable Redis
service and fails if the service is unavailable.  No MongoDB audit write is
needed to prove the Redis contract, so the audit boundary is replaced with a
small async no-op in this test module.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.features.auth import repository as repository_module
from app.features.auth.exceptions import AuthInfrastructureError
from app.features.auth.repository import RefreshTokenRepository, SessionData

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.integration


async def _noop_audit_insert(_audit: object) -> None:
    return None


class _NoopAuditQuery:
    async def update(self, _update: object) -> None:
        return None


def _noop_audit_find(*_args: object, **_kwargs: object) -> _NoopAuditQuery:
    return _NoopAuditQuery()


@pytest.fixture
async def live_redis() -> AsyncIterator[Redis]:
    url = os.getenv("LIVE_REDIS_URL")
    if not url:
        pytest.skip("LIVE_REDIS_URL is required; start the documented disposable Redis service")

    client = Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        retry_on_timeout=False,
    )
    try:
        await client.ping()
    except RedisError as exc:
        await client.aclose()
        pytest.fail(f"LIVE_REDIS_URL is configured but Redis is unavailable: {exc}")
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest.fixture
def session_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repository_module.TokenAuditLog, "insert", _noop_audit_insert)
    monkeypatch.setattr(repository_module.TokenAuditLog, "find", _noop_audit_find)


def _session(user_id: str, *, ttl: int = 120) -> SessionData:
    now = datetime.now(UTC)
    return SessionData(
        session_id=f"live-{uuid4()}",
        user_id=user_id,
        device_id="ci-device",
        created_at=now,
        expires_at=now + timedelta(seconds=ttl),
        ttl=ttl,
        authz_revision=4,
    )


@pytest.mark.asyncio
async def test_real_redis_store_revoke_all_and_index_ttl(
    live_redis: Redis,
    session_store: None,
) -> None:
    store = RefreshTokenRepository(live_redis)
    first = _session("live-user", ttl=120)
    second = _session("live-user", ttl=30)

    assert isinstance(await store.store_session(first), Success)
    assert isinstance(await store.store_session(second), Success)
    before = await live_redis.ttl("auth:user_sessions:live-user")
    assert before > 0

    result = await store.revoke_all_user_sessions("live-user", authz_revision=5)

    assert result == Success([first.session_id, second.session_id]) or result == Success(
        [second.session_id, first.session_id]
    )
    assert await live_redis.exists(f"auth:session:{first.session_id}") == 0
    assert await live_redis.exists(f"auth:session:{second.session_id}") == 0
    assert await live_redis.exists("auth:user_sessions:live-user") == 0
    assert await live_redis.get("auth:user_authz_revision:live-user") == "5"
    assert before >= 1


@pytest.mark.asyncio
async def test_real_redis_refusal_is_typed_and_recovery_succeeds(
    live_redis: Redis,
    session_store: None,
) -> None:
    refused = Redis.from_url(
        "redis://127.0.0.1:1/15",
        decode_responses=True,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
        retry_on_timeout=False,
    )
    try:
        failed = await RefreshTokenRepository(refused).get_session("missing")
    finally:
        await refused.aclose()

    assert isinstance(failed, Failure)
    assert isinstance(failed.failure(), AuthInfrastructureError)
    assert failed.failure().retryable is True

    recovered = RefreshTokenRepository(live_redis)
    session = _session("recovered-user")
    assert isinstance(await recovered.store_session(session), Success)
    assert await recovered.get_session(session.session_id) == Success(session)


@pytest.mark.asyncio
async def test_real_redis_timeout_and_client_reconnect_recover(
    live_redis: Redis,
    session_store: None,
) -> None:
    session = _session("timeout-user")
    store = RefreshTokenRepository(live_redis)
    assert isinstance(await store.store_session(session), Success)

    control = Redis.from_url(os.environ["LIVE_REDIS_URL"], decode_responses=True)
    timed = Redis.from_url(
        os.environ["LIVE_REDIS_URL"],
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=0.01,
        retry_on_timeout=False,
    )
    try:
        await control.execute_command("CLIENT", "PAUSE", 200, "ALL")
        timed_result = await RefreshTokenRepository(timed).get_session(session.session_id)
    finally:
        await timed.aclose()
        await asyncio.sleep(0.25)
        await control.aclose()

    assert isinstance(timed_result, Failure)
    assert isinstance(timed_result.failure(), AuthInfrastructureError)
    assert timed_result.failure().retryable is True

    # A new client represents a worker reconnecting after the outage.  The
    # session remains authoritative and is readable after Redis recovers.
    reconnected = Redis.from_url(os.environ["LIVE_REDIS_URL"], decode_responses=True)
    try:
        assert await RefreshTokenRepository(reconnected).get_session(session.session_id) == Success(
            session
        )
    finally:
        await reconnected.aclose()
