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
from pyrate_limiter import Limiter, Rate
from pyrate_limiter.buckets import InMemoryBucket
from redis.asyncio import Redis
from redis.exceptions import RedisError
from returns.result import Failure, Success

from app.features.auth import repository as repository_module
from app.features.auth.dependencies import get_token_claims
from app.features.auth.exceptions import AuthAuthenticationError, AuthInfrastructureError
from app.features.auth.models import UserRole
from app.features.auth.pending_revocation import (
    PendingRevocationConfig,
    PendingRevocationReconciler,
)
from app.features.auth.repository import RefreshTokenRepository, SessionData
from app.features.auth.websocket_revocation import (
    REVOCATION_STREAM_KEY,
    new_event,
    publish_event,
)
from app.features.auth.websocket_security import WebSocketSecurityService
from app.features.users.exceptions import UsersInfrastructureError
from app.features.users.service import UserAdminService
from app.security.tokens import TokenClaims, derive_refresh_token_material
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


class _WebSocketProbe:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.close_count = 0

    async def send_json(self, _payload: object) -> None:
        return None

    async def close(self, **_kwargs: object) -> None:
        self.close_count += 1
        self.closed.set()


def _revocation_service(redis: Redis) -> WebSocketSecurityService:
    settings = SimpleNamespace(
        WEBSOCKET_REVOCATION_ENABLED=True,
        WEBSOCKET_REVOCATION_BATCH_SIZE=32,
        WEBSOCKET_REVOCATION_RECONNECT_BASE_SECONDS=0.01,
        WEBSOCKET_REVOCATION_RECONNECT_MAX_SECONDS=0.05,
        WEBSOCKET_PRESENCE_TTL_SECONDS=360,
    )
    limiter = Limiter(InMemoryBucket([Rate(1000, 1000)]))
    return WebSocketSecurityService(
        redis=redis,
        settings=settings,
        user_limiter=limiter,
        connection_limiter=Limiter(InMemoryBucket([Rate(1000, 1000)])),
    )


async def _seed_stream_cursor(redis: Redis) -> str:
    return str(
        await redis.xadd(
            REVOCATION_STREAM_KEY,
            {"version": "1", "event_id": f"marker-{uuid4().hex}"},
        )
    )


async def test_distributed_revocation_reaches_two_workers_and_scopes_connections(
    redis_harness: _RedisHarness,
) -> None:
    """Two independent listeners close only their matching local sockets."""
    redis = redis_harness.client
    worker_a = _revocation_service(redis)
    worker_b = _revocation_service(redis)
    marker = await _seed_stream_cursor(redis)
    worker_a._revocation_last_id = marker
    worker_b._revocation_last_id = marker

    user_id = _user_id()
    target_a = _WebSocketProbe()
    unrelated_a = _WebSocketProbe()
    wildcard_target = _WebSocketProbe()
    target_b = _WebSocketProbe()
    claims_a = _claims(user_id=user_id, session_id="session-target", revision=0)
    claims_other = _claims(user_id=user_id, session_id="session-other", revision=0)
    worker_a._live_connections["a-target"] = (
        target_a,
        worker_a.build_context(claims=claims_a, origin=None, connection_id="a-target"),
    )
    worker_a._live_connections["a-other"] = (
        unrelated_a,
        worker_a.build_context(claims=claims_other, origin=None, connection_id="a-other"),
    )
    claims_wildcard = _claims(user_id=user_id, session_id="session-wildcard", revision=0)
    worker_a._live_connections["a-wildcard"] = (
        wildcard_target,
        worker_a.build_context(
            claims=claims_wildcard,
            origin=None,
            connection_id="a-wildcard",
        ),
    )
    worker_b._live_connections["b-target"] = (
        target_b,
        worker_b.build_context(claims=claims_a, origin=None, connection_id="b-target"),
    )
    listeners = [
        asyncio.create_task(worker_a.run_revocation_listener()),
        asyncio.create_task(worker_b.run_revocation_listener()),
    ]
    try:
        await asyncio.gather(
            worker_a._revocation_cursor_ready.wait(),
            worker_b._revocation_cursor_ready.wait(),
        )
        await publish_event(
            redis,
            new_event(
                scope="session",
                user_id=user_id,
                session_id="session-target",
                reason="role_changed",
            ),
        )
        await asyncio.wait_for(
            asyncio.gather(target_a.closed.wait(), target_b.closed.wait()), timeout=2
        )
        assert target_a.close_count == target_b.close_count == 1
        assert not unrelated_a.closed.is_set()
        await worker_a._handle_revocation_event(
            new_event(
                scope="user",
                user_id=user_id,
                except_session_id="session-other",
                reason="deactivation",
            )
        )
        assert wildcard_target.close_count == 1
        assert not unrelated_a.closed.is_set()

        revision_probe = _WebSocketProbe()
        revision_claims = _claims(
            user_id=user_id,
            session_id="session-revision-excluded",
            revision=2,
        )
        worker_a._live_connections["a-revision"] = (
            revision_probe,
            worker_a.build_context(
                claims=revision_claims,
                origin=None,
                connection_id="a-revision",
            ),
        )
        await worker_a._handle_revocation_event(
            new_event(
                scope="user",
                user_id=user_id,
                except_session_id="session-revision-excluded",
                authz_revision=2,
                reason="role_changed",
            )
        )
        await worker_a._handle_revocation_event(
            new_event(
                scope="user",
                user_id=user_id,
                authz_revision=1,
                reason="stale_role_change",
            )
        )
        assert not revision_probe.closed.is_set()
        await worker_a._handle_revocation_event(
            new_event(
                scope="user",
                user_id=user_id,
                authz_revision=3,
                reason="new_role_change",
            )
        )
        assert revision_probe.close_count == 1
    finally:
        for listener in listeners:
            listener.cancel()
        await asyncio.gather(*listeners, return_exceptions=True)
        worker_a.close()
        worker_b.close()


async def test_distributed_revocation_rejects_spoofed_and_malformed_events_and_deduplicates(
    redis_harness: _RedisHarness,
) -> None:
    redis = redis_harness.client
    worker = _revocation_service(redis)
    marker = await _seed_stream_cursor(redis)
    worker._revocation_last_id = marker
    user_id = _user_id()
    probe = _WebSocketProbe()
    claims = _claims(user_id=user_id, session_id="session-valid", revision=0)
    worker._live_connections["connection"] = (
        probe,
        worker.build_context(claims=claims, origin=None, connection_id="connection"),
    )
    listeners = [asyncio.create_task(worker.run_revocation_listener())]
    try:
        await worker._revocation_cursor_ready.wait()
        await redis.xadd(REVOCATION_STREAM_KEY, {"version": "1", "scope": "session"})
        spoofed = new_event(
            scope="session",
            user_id="another-user",
            session_id="session-valid",
            reason="spoof",
        )
        for _ in range(2):
            worker._revocation_activity.clear()
            await publish_event(redis, spoofed)
            await asyncio.wait_for(worker._revocation_activity.wait(), timeout=2)
        worker._revocation_activity.clear()
        assert not probe.closed.is_set()

        valid = new_event(
            scope="session",
            user_id=user_id,
            session_id="session-valid",
            reason="logout",
        )
        worker._revocation_activity.clear()
        await publish_event(redis, valid)
        await asyncio.wait_for(probe.closed.wait(), timeout=2)
        assert probe.close_count == 1

        duplicate_probe = _WebSocketProbe()
        worker._live_connections["connection-duplicate"] = (
            duplicate_probe,
            worker.build_context(
                claims=claims,
                origin=None,
                connection_id="connection-duplicate",
            ),
        )
        worker._revocation_activity.clear()
        await publish_event(redis, valid)
        await asyncio.wait_for(worker._revocation_activity.wait(), timeout=2)
        assert duplicate_probe.close_count == 0
    finally:
        for listener in listeners:
            listener.cancel()
        await asyncio.gather(*listeners, return_exceptions=True)
        worker.close()


async def test_distributed_listener_reconnects_after_redis_connection_drop(
    redis_harness: _RedisHarness,
) -> None:
    redis = redis_harness.client
    worker = _revocation_service(redis)
    marker = await _seed_stream_cursor(redis)
    worker._revocation_last_id = marker
    user_id = _user_id()
    probe = _WebSocketProbe()
    claims = _claims(user_id=user_id, session_id="session-reconnect", revision=0)
    worker._live_connections["reconnect"] = (
        probe,
        worker.build_context(claims=claims, origin=None, connection_id="reconnect"),
    )
    listener = asyncio.create_task(worker.run_revocation_listener())
    try:
        await worker._revocation_cursor_ready.wait()
        # Drop the listener's active blocking connection. It must reconnect
        # and continue from the cursor without relying on a new subscription.
        await redis.connection_pool.disconnect()
        await publish_event(
            redis,
            new_event(
                scope="session",
                user_id=user_id,
                session_id="session-reconnect",
                reason="outage_recovery",
            ),
        )
        await asyncio.wait_for(probe.closed.wait(), timeout=3)
        assert probe.close_count == 1
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)
        worker.close()


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
        for session in sessions:
            assert await store.get_session(session.session_id) == Success(None)


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
    store, redis = repository
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
        index_ids = await redis.client.smembers(store._user_sessions_key(user_id))
        fresh_result = await store.get_session(fresh.session_id)
        assert (fresh_result == Success(fresh)) is (fresh.session_id in index_ids)
        assert index_ids in (set(), {fresh.session_id})
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
            assert (revision == "0" and ids and all(value is not None for value in values)) or (
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
    repository: tuple[RefreshTokenRepository, _RedisHarness],
) -> None:
    recovered_store, redis_harness = repository
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

    recovered_service = UserAdminService(UserRepo(), recovered_store)
    recovered = await recovered_service.update_role(user_id, UserRole.MODERATOR, "admin-id")
    assert isinstance(recovered, Success)
    assert user.session_revocation_pending is False
    assert await redis_harness.client.get(recovered_store._authz_revision_key(user_id)) == "2"


@pytest.mark.asyncio
async def test_real_redis_refresh_rotation_replay_and_grace(
    repository: tuple[RefreshTokenRepository, _RedisHarness],
) -> None:
    """Actual Redis proves CAS rotation, bounded grace, and post-grace replay."""
    store, harness = repository
    user_id = _user_id()
    family_id = f"family-{uuid4().hex}"
    session = _session(user_id, revision=4)
    session = session.model_copy(update={"family_id": family_id})
    token_id, token_secret = derive_refresh_token_material(family_id, 0)
    child_id, child_secret = derive_refresh_token_material(family_id, 1)
    assert isinstance(await store.store_session(session), Success)
    assert isinstance(
        await store.create_refresh_family(
            family_id=family_id,
            user_id=user_id,
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

    results = await asyncio.gather(
        *(
            store.rotate_refresh_token(
                family_id=family_id,
                token_id=token_id,
                token_secret=token_secret,
                child_token_id=child_id,
                child_token_secret=child_secret,
                child_generation=1,
                expected_authz_revision=session.authz_revision,
                now=session.created_at,
                grace_seconds=10,
            )
            for _ in range(2)
        )
    )
    outcomes = [result.unwrap() for result in results]
    assert sorted(outcome.status for outcome in outcomes) == ["grace", "rotated"]
    assert {outcome.generation for outcome in outcomes} == {1}

    replay = await store.rotate_refresh_token(
        family_id=family_id,
        token_id=token_id,
        token_secret=token_secret,
        child_token_id=child_id,
        child_token_secret=child_secret,
        child_generation=1,
        expected_authz_revision=session.authz_revision,
        now=session.created_at + timedelta(seconds=11),
        grace_seconds=10,
    )
    assert isinstance(replay, Success)
    assert replay.unwrap().status == "reused"
    family_key = f"{store._key_prefix}auth:refresh_family:{family_id}"
    assert await harness.client.hget(family_key, "status") == "revoked"
    assert await harness.client.exists(store._session_key(session.session_id)) == 0


async def test_pending_revocation_lease_and_retry_use_real_redis(
    redis_harness: _RedisHarness,
) -> None:
    """A real Redis server fences duplicate workers and retains retry state."""
    user = SimpleNamespace(
        id=_user_id(),
        role_revision=7,
        session_revocation_pending=True,
        updated_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    attempts = 0

    async def list_pending(_limit: int, _offset: int) -> list[SimpleNamespace]:
        return [user]

    async def get_user(_user_id: str) -> SimpleNamespace:
        return user

    async def revoke(_user_id: str, _revision: int):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return Failure(AuthInfrastructureError(message="Redis outage"))
        return Success(user)

    async def clear(current: SimpleNamespace, _revision: int):
        current.session_revocation_pending = False
        return Success(current)

    config = PendingRevocationConfig(
        backoff_base_seconds=60,
        backoff_max_seconds=60,
        jitter_seconds=0,
        lease_seconds=30,
    )
    first = PendingRevocationReconciler(
        redis_harness.client,
        list_pending=list_pending,
        get_user=get_user,
        revoke_sessions=revoke,
        clear_marker=clear,
        config=config,
        monotonic=lambda: 0.0,
        random=lambda _upper: 0,
    )
    second = PendingRevocationReconciler(
        redis_harness.client,
        list_pending=list_pending,
        get_user=get_user,
        revoke_sessions=revoke,
        clear_marker=clear,
        config=config,
        monotonic=lambda: 0.0,
        random=lambda _upper: 0,
    )

    failed = await first.run()
    deferred = await second.run()
    recovered = await second.run(force=True)

    assert failed["retryable_failure"] == 1
    assert deferred["skipped"] == 1
    assert recovered["succeeded"] == 1
    assert attempts == 2
    assert user.session_revocation_pending is False
