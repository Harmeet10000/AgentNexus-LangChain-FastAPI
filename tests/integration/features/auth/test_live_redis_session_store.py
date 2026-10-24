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
    from collections.abc import AsyncIterator, Callable

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
        await client.aclose()


@pytest.fixture
async def session_store(
    live_redis: Redis,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[[SessionData], None]]:
    monkeypatch.setattr(repository_module.TokenAuditLog, "insert", _noop_audit_insert)
    monkeypatch.setattr(repository_module.TokenAuditLog, "find", _noop_audit_find)
    created: list[SessionData] = []
    yield created.append
    for session in created:
        await live_redis.delete(f"auth:session:{session.session_id}")
        await live_redis.srem(f"auth:user_sessions:{session.user_id}", session.session_id)
        await live_redis.delete(f"auth:user_authz_revision:{session.user_id}")


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


def _user_id() -> str:
    return f"live-user-{uuid4()}"


@pytest.mark.asyncio
async def test_real_redis_store_revoke_all_and_index_ttl(
    live_redis: Redis,
    session_store: Callable[[SessionData], None],
) -> None:
    store = RefreshTokenRepository(live_redis)
    user_id = _user_id()
    first = _session(user_id, ttl=120)
    second = _session(user_id, ttl=30)

    assert isinstance(await store.store_session(first), Success)
    assert isinstance(await store.store_session(second), Success)
    session_store(first)
    session_store(second)
    before = await live_redis.ttl(f"auth:user_sessions:{user_id}")
    assert before > 0
    assert before >= 110

    result = await store.revoke_all_user_sessions(user_id, authz_revision=5)

    assert result == Success([first.session_id, second.session_id]) or result == Success(
        [second.session_id, first.session_id]
    )
    assert await live_redis.exists(f"auth:session:{first.session_id}") == 0
    assert await live_redis.exists(f"auth:session:{second.session_id}") == 0
    assert await live_redis.exists(f"auth:user_sessions:{user_id}") == 0
    assert await live_redis.get(f"auth:user_authz_revision:{user_id}") == "5"


@pytest.mark.asyncio
async def test_real_redis_refusal_is_typed_and_recovery_succeeds(
    live_redis: Redis,
    session_store: Callable[[SessionData], None],
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
    session = _session(_user_id())
    assert isinstance(await recovered.store_session(session), Success)
    session_store(session)
    assert await recovered.get_session(session.session_id) == Success(session)


@pytest.mark.asyncio
async def test_real_redis_timeout_and_client_reconnect_recover(
    live_redis: Redis,
    session_store: Callable[[SessionData], None],
) -> None:
    session = _session(_user_id())
    store = RefreshTokenRepository(live_redis)
    assert isinstance(await store.store_session(session), Success)
    session_store(session)

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


@pytest.mark.asyncio
async def test_real_redis_session_survives_service_restart(
    live_redis: Redis,
    session_store: Callable[[SessionData], None],
) -> None:
    if os.getenv("LIVE_REDIS_ALLOW_DOCKER_RESTART") != "1":
        pytest.skip("LIVE_REDIS_ALLOW_DOCKER_RESTART=1 is required for the CI-only restart test")

    list_process = await asyncio.create_subprocess_exec(
        "/usr/bin/docker",
        "ps",
        "--filter",
        "ancestor=redis:7.4.2-alpine",
        "--format",
        "{{.ID}}",
        stdout=asyncio.subprocess.PIPE,
    )
    list_stdout, _ = await list_process.communicate()
    if list_process.returncode != 0 or list_stdout is None:
        pytest.fail("Unable to locate the disposable Redis service container")
    container_ids = list_stdout.decode().strip().splitlines()
    if not container_ids:
        pytest.fail("Unable to locate the disposable Redis service container")
    container_id = container_ids[0]
    session = _session(_user_id())
    assert isinstance(await RefreshTokenRepository(live_redis).store_session(session), Success)
    session_store(session)

    restart_process = await asyncio.create_subprocess_exec(
        "/usr/bin/docker",
        "restart",
        container_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await restart_process.communicate()
    if restart_process.returncode != 0:
        pytest.fail("Unable to restart the disposable Redis service container")
    store = RefreshTokenRepository(live_redis)
    for _ in range(30):
        await asyncio.sleep(1)
        recovered = await store.get_session(session.session_id)
        if recovered == Success(session):
            break
    else:
        pytest.fail("Redis session was not readable after the service restart")
