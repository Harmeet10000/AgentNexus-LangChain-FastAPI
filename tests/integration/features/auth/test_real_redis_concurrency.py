"""Deterministic concurrency proofs for the Redis-backed session authority.

Run locally with ``uv run pytest tests/integration/features/auth/test_real_redis_concurrency.py
-m real_redis -v``.  The fixture starts an unprivileged, disposable ``redis-server`` when
``REDIS_SERVER_BIN`` (or ``redis-server`` on ``PATH``) is available.  Set
``REAL_REDIS_URL`` to use an already-running disposable Redis instead.  Every repository
instance receives a unique key prefix, so the suite never flushes a shared database.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.features.auth import repository as repository_module
from app.features.auth.dependencies import get_token_claims
from app.features.auth.exceptions import AuthAuthenticationError
from app.features.auth.models import UserRole
from app.features.auth.repository import RefreshTokenRepository, SessionData
from app.features.users.exceptions import UsersInfrastructureError
from app.features.users.service import UserAdminService
from app.security.tokens import TokenClaims
from app.utils import UnauthorizedException

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

pytestmark = [pytest.mark.integration, pytest.mark.real_redis]

_OBSERVE_STATE_SCRIPT = """
local revision = redis.call('GET', KEYS[2])
local ids = redis.call('SMEMBERS', KEYS[1])
local values = {}
for _, session_id in ipairs(ids) do
    table.insert(values, redis.call('GET', ARGV[1] .. session_id))
end
return {revision, ids, values}
"""


@dataclass(frozen=True)
class _RedisHarness:
    client: Redis
    url: str


class _NoopAuditQuery:
    async def update(self, _update: object) -> None:
        return None


def _noop_audit_find(*_args: object, **_kwargs: object) -> _NoopAuditQuery:
    return _NoopAuditQuery()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_redis(client: Redis, process: asyncio.subprocess.Process | None) -> None:
    for _ in range(200):
        if process is not None and process.returncode is not None:
            return_code = process.returncode
            message = f"redis-server exited before readiness: {return_code}"
            raise RuntimeError(message)
        try:
            await client.ping()
        except RedisError:
            await asyncio.sleep(0.01)
        else:
            return
    message = "redis-server did not become ready"
    raise RuntimeError(message)


@pytest.fixture
async def redis_harness() -> AsyncIterator[_RedisHarness]:
    configured_url = os.getenv("REAL_REDIS_URL") or os.getenv("LIVE_REDIS_URL")
    process: asyncio.subprocess.Process | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    if configured_url:
        url = configured_url
    else:
        redis_server = os.getenv("REDIS_SERVER_BIN") or shutil.which("redis-server")
        if redis_server is None:
            pytest.skip(
                "real Redis unavailable: set REAL_REDIS_URL or install redis-server and set "
                "REDIS_SERVER_BIN"
            )
        temp_dir = tempfile.TemporaryDirectory(prefix="agentnexus-redis-")
        port = _free_port()
        url = f"redis://127.0.0.1:{port}/0"
        process = await asyncio.create_subprocess_exec(
            redis_server,
            "--bind",
            "127.0.0.1",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            temp_dir.name,
            "--protected-mode",
            "yes",
            "--loglevel",
            "warning",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    client = Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        retry_on_timeout=False,
    )
    try:
        await _wait_for_redis(client, process)
        yield _RedisHarness(client=client, url=url)
    finally:
        await client.aclose()
        if process is not None:
            process.terminate()
            await process.wait()
        if temp_dir is not None:
            temp_dir.cleanup()


@pytest.fixture
async def repository(
    redis_harness: _RedisHarness, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[RefreshTokenRepository, _RedisHarness]]:
    monkeypatch.setattr(repository_module.TokenAuditLog, "insert", _noop_audit_insert)
    monkeypatch.setattr(repository_module.TokenAuditLog, "find", _noop_audit_find)
    prefix = f"test:{uuid4().hex}:"
    yield RefreshTokenRepository(redis_harness.client, key_prefix=prefix), redis_harness
    keys = [key async for key in redis_harness.client.scan_iter(match=f"{prefix}*")]
    if keys:
        await redis_harness.client.delete(*keys)


async def _noop_audit_insert(_audit: object) -> None:
    return None


def _session(
    user_id: str,
    *,
    revision: int = 0,
    ttl: int = 120,
    session_id: str | None = None,
) -> SessionData:
    now = datetime.now(UTC)
    return SessionData(
        session_id=session_id or f"session-{uuid4().hex}",
        user_id=user_id,
        device_id="real-redis-test",
        created_at=now,
        expires_at=now + timedelta(seconds=ttl),
        ttl=ttl,
        authz_revision=revision,
    )


def _user_id() -> str:
    return uuid4().hex[:24]


def _claims(*, user_id: str, session_id: str, revision: int) -> TokenClaims:
    return TokenClaims(
        sub=user_id,
        jti=f"jti-{session_id}",
        sid=session_id,
        role="user",
        permissions=[],
        token_type="access",
        authz_revision=revision,
    )


async def _rendezvous[T](barrier: asyncio.Barrier, operation: Awaitable[T]) -> T:
    await barrier.wait()
    return await operation


async def test_concurrent_creation_and_role_revoke_has_no_stale_survivor(
    repository: tuple[RefreshTokenRepository, _RedisHarness],
) -> None:
    store, redis = repository
    rounds = int(os.getenv("REAL_REDIS_RACE_ROUNDS", "16"))

    for _ in range(rounds):
        user_id = _user_id()
        sessions = [_session(user_id) for _ in range(8)]
        barrier = asyncio.Barrier(len(sessions) + 1)
        tasks = [
            asyncio.create_task(_rendezvous(barrier, store.store_session(session)))
            for session in sessions
        ]
        tasks.append(
            asyncio.create_task(
                _rendezvous(
                    barrier,
                    store.revoke_all_user_sessions(user_id, authz_revision=1),
                )
            )
        )
        results = await asyncio.gather(*tasks)

        assert all(isinstance(result, (Success, Failure)) for result in results)
        assert await redis.client.get(store._authz_revision_key(user_id)) == "1"
        assert await store.get_user_sessions(user_id) == Success([])


async def test_overlapping_role_revisions_are_monotonic(
    repository: tuple[RefreshTokenRepository, _RedisHarness],
) -> None:
    store, redis = repository
    rounds = int(os.getenv("REAL_REDIS_RACE_ROUNDS", "16"))

    for _ in range(rounds):
        user_id = _user_id()
        old = _session(user_id)
        assert await store.store_session(old) == Success(None)
        barrier = asyncio.Barrier(2)
        lower = asyncio.create_task(
            _rendezvous(
                barrier,
                store.revoke_all_user_sessions(user_id, authz_revision=1),
            )
        )
        higher = asyncio.create_task(
            _rendezvous(
                barrier,
                store.revoke_all_user_sessions(user_id, authz_revision=3),
            )
        )
        await asyncio.gather(lower, higher)

        assert await redis.client.get(store._authz_revision_key(user_id)) == "3"
        assert await store.get_user_sessions(user_id) == Success([])
        stale = await store.store_session(_session(user_id, revision=2))
        assert isinstance(stale, Failure)
        assert isinstance(stale.failure(), AuthAuthenticationError)
        assert await store.store_session(_session(user_id, revision=3)) == Success(None)


async def test_revoke_all_and_current_epoch_login_have_no_omitted_index_entry(
    repository: tuple[RefreshTokenRepository, _RedisHarness],
) -> None:
    store, _redis = repository
    rounds = int(os.getenv("REAL_REDIS_RACE_ROUNDS", "16"))

    for _ in range(rounds):
        user_id = _user_id()
        assert await store.store_session(_session(user_id)) == Success(None)
        fresh = _session(user_id, revision=1)
        barrier = asyncio.Barrier(2)
        revoke = asyncio.create_task(
            _rendezvous(
                barrier,
                store.revoke_all_user_sessions(user_id, authz_revision=1),
            )
        )
        create = asyncio.create_task(_rendezvous(barrier, store.store_session(fresh)))
        await asyncio.gather(revoke, create)

        listed = await store.get_user_sessions(user_id)
        assert isinstance(listed, Success)
        sessions = listed.unwrap()
        assert all(session.session_id == fresh.session_id for session in sessions)
        assert all(session.authz_revision == 1 for session in sessions)
        for session in sessions:
            assert await store.get_session(session.session_id) == Success(session)


async def test_multiple_clients_observe_atomic_revoke_state(
    repository: tuple[RefreshTokenRepository, _RedisHarness],
) -> None:
    store, harness = repository
    observer = Redis.from_url(harness.url, decode_responses=True)
    try:
        user_id = _user_id()
        sessions = [_session(user_id) for _ in range(12)]
        for session in sessions:
            assert await store.store_session(session) == Success(None)

        barrier = asyncio.Barrier(2)

        async def revoke() -> object:
            await barrier.wait()
            return await store.revoke_all_user_sessions(user_id, authz_revision=1)

        async def observe() -> list[tuple[set[str], list[str | None], str | None]]:
            await barrier.wait()
            observations: list[tuple[set[str], list[str | None], str | None]] = []
            for _ in range(24):
                revision, ids, values = await observer.eval(
                    _OBSERVE_STATE_SCRIPT,
                    2,
                    store._user_sessions_key(user_id),
                    store._authz_revision_key(user_id),
                    store._session_key_prefix(),
                )
                observations.append((ids, values, revision))
            return observations

        _, observations = await asyncio.gather(revoke(), observe())
        for ids, values, revision in observations:
            assert (revision is None and ids and all(value is not None for value in values)) or (
                revision == "1" and not ids and not values
            )
    finally:
        await observer.aclose()


async def test_real_redis_rejects_malformed_expired_owner_and_epoch_records(
    repository: tuple[RefreshTokenRepository, _RedisHarness],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, redis = repository
    user_id = _user_id()
    session = _session(user_id, revision=4, session_id="validation-session")
    assert await store.store_session(session) == Success(None)

    monkeypatch.setattr(
        "app.features.auth.dependencies.decode_token",
        lambda _token: _claims(user_id="other-user", session_id=session.session_id, revision=4),
    )
    with pytest.raises(UnauthorizedException, match="Session expired or revoked"):
        await get_token_claims("raw", store)

    monkeypatch.setattr(
        "app.features.auth.dependencies.decode_token",
        lambda _token: _claims(user_id=user_id, session_id=session.session_id, revision=3),
    )
    with pytest.raises(UnauthorizedException, match="Session expired or revoked"):
        await get_token_claims("raw", store)

    expired = session.model_copy(update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)})
    await redis.client.set(store._session_key(session.session_id), expired.model_dump_json())
    monkeypatch.setattr(
        "app.features.auth.dependencies.decode_token",
        lambda _token: _claims(user_id=user_id, session_id=session.session_id, revision=4),
    )
    with pytest.raises(UnauthorizedException, match="Session expired or revoked"):
        await get_token_claims("raw", store)

    await redis.client.set(store._session_key("malformed"), "not-json")
    malformed = await store.get_session("malformed")
    assert isinstance(malformed, Failure)
    await redis.client.set(store._session_key("expired"), "not-json")
    await redis.client.expire(store._session_key("expired"), 0)
    assert await store.get_session("expired") == Success(None)


async def test_session_index_ttl_and_expired_member_repair_use_real_redis(
    repository: tuple[RefreshTokenRepository, _RedisHarness],
) -> None:
    store, redis = repository
    user_id = _user_id()
    long_session = _session(user_id, ttl=120)
    short_session = _session(user_id, ttl=30)
    assert await store.store_session(long_session) == Success(None)
    before = await redis.client.ttl(store._user_sessions_key(user_id))
    assert before >= 110
    assert await store.store_session(short_session) == Success(None)
    assert await redis.client.ttl(store._user_sessions_key(user_id)) > 30

    await redis.client.delete(store._session_key(long_session.session_id))
    listed = await store.get_user_sessions(user_id)
    assert listed == Success([short_session])
    assert await redis.client.smembers(store._user_sessions_key(user_id)) == {
        short_session.session_id
    }


async def test_pending_marker_survives_redis_outage_and_clears_after_recovery(
    redis_harness: _RedisHarness,
) -> None:
    user_id = _user_id()
    now = datetime.now(UTC)
    user = SimpleNamespace(
        id=user_id,
        email=f"{user_id}@example.test",
        full_name=None,
        role=UserRole.ADMIN,
        is_verified=True,
        is_active=True,
        created_at=now,
        updated_at=now,
        oauth_accounts=[],
        role_revision=1,
        session_revocation_pending=True,
    )

    class UserRepo:
        async def find_by_id(self, _user_id: str) -> Success[SimpleNamespace]:
            return Success(user)

        async def update_role(
            self, _user: SimpleNamespace, role: UserRole
        ) -> Success[SimpleNamespace]:
            user.role = role
            user.role_revision += 1
            user.session_revocation_pending = True
            return Success(user)

        async def clear_session_revocation_pending(
            self, _user: SimpleNamespace, *, expected_revision: int
        ) -> Success[SimpleNamespace]:
            if expected_revision == user.role_revision:
                user.session_revocation_pending = False
            return Success(user)

    refused = Redis.from_url(
        "redis://127.0.0.1:1/0",
        decode_responses=True,
        socket_connect_timeout=0.1,
        socket_timeout=0.1,
        retry_on_timeout=False,
    )
    try:
        failed_service = UserAdminService(UserRepo(), RefreshTokenRepository(refused))
        failed = await failed_service.update_role(user_id, UserRole.MODERATOR, "admin-id")
    finally:
        await refused.aclose()
    assert isinstance(failed, Failure)
    assert isinstance(failed.failure(), UsersInfrastructureError)
    assert failed.failure().retryable is True
    assert user.session_revocation_pending is True

    recovered_store = RefreshTokenRepository(
        redis_harness.client, key_prefix=f"test:{uuid4().hex}:"
    )
    recovered_service = UserAdminService(UserRepo(), recovered_store)
    recovered = await recovered_service.update_role(user_id, UserRole.MODERATOR, "admin-id")
    assert isinstance(recovered, Success)
    assert user.session_revocation_pending is False
    assert await redis_harness.client.get(recovered_store._authz_revision_key(user_id)) == "2"
