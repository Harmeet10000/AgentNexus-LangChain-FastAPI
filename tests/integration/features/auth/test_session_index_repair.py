"""Real-Redis coverage for the bounded session-index migration."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import tempfile
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError
from returns.result import Success

from app.features.auth import repository as repository_module
from app.features.auth.repository import RefreshTokenRepository, SessionData
from app.features.auth.session_index_repair import SessionIndexRepair

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = [pytest.mark.integration, pytest.mark.real_redis]


class _NoopAuditQuery:
    async def update(self, _update: object) -> None:
        return None


async def _noop_audit_insert(_audit: object) -> None:
    return None


def _noop_audit_find(*_args: object, **_kwargs: object) -> _NoopAuditQuery:
    return _NoopAuditQuery()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
async def redis_client() -> AsyncIterator[tuple[Redis, str]]:
    configured_url = os.getenv("REAL_REDIS_URL") or os.getenv("LIVE_REDIS_URL")
    process: asyncio.subprocess.Process | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if configured_url:
        url = configured_url
    else:
        redis_server = os.getenv("REDIS_SERVER_BIN") or shutil.which("redis-server")
        if redis_server is None:
            pytest.skip("real Redis unavailable; set REAL_REDIS_URL or install redis-server")
        temp_dir = tempfile.TemporaryDirectory(prefix="agentnexus-index-repair-")
        url = f"redis://127.0.0.1:{_free_port()}/0"
        process = await asyncio.create_subprocess_exec(
            redis_server,
            "--bind",
            "127.0.0.1",
            "--port",
            url.rsplit(":", 1)[-1].split("/", 1)[0],
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
    client = Redis.from_url(url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
    try:
        for _ in range(200):
            try:
                await client.ping()
                break
            except RedisError:
                await asyncio.sleep(0.01)
        else:
            pytest.fail("Redis did not become ready")
        prefix = f"repair:{uuid4().hex}:"
        yield client, prefix
    finally:
        await client.aclose()
        if process is not None:
            process.terminate()
            await process.wait()
        if temp_dir is not None:
            temp_dir.cleanup()


def _session(user_id: str, *, ttl: int = 180, session_id: str | None = None) -> SessionData:
    now = datetime.now(UTC)
    return SessionData(
        session_id=session_id or f"session-{uuid4().hex}",
        user_id=user_id,
        device_id="repair-test",
        created_at=now,
        expires_at=now + timedelta(seconds=ttl),
        ttl=ttl,
    )


def _user_id() -> str:
    return uuid4().hex[:24]


async def _keys(client: Redis, prefix: str) -> list[str]:
    return [key async for key in client.scan_iter(match=f"{prefix}*")]


@pytest.fixture
async def repair_context(
    redis_client: tuple[Redis, str], monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Redis, str, RefreshTokenRepository, SessionIndexRepair]]:
    client, prefix = redis_client
    monkeypatch.setattr(repository_module.TokenAuditLog, "insert", _noop_audit_insert)
    monkeypatch.setattr(repository_module.TokenAuditLog, "find", _noop_audit_find)
    store = RefreshTokenRepository(client, key_prefix=prefix)
    yield client, prefix, store, SessionIndexRepair(client, key_prefix=prefix, scan_count=10)
    keys = await _keys(client, prefix)
    if keys:
        await client.delete(*keys)


async def test_repairs_short_ttl_without_changing_session_ttl(
    repair_context: tuple[Redis, str, RefreshTokenRepository, SessionIndexRepair],
) -> None:
    client, _prefix, store, repair = repair_context
    session = _session(_user_id(), ttl=180)
    assert isinstance(await store.store_session(session), Success)
    index_key = store._user_sessions_key(session.user_id)
    session_key = store._session_key(session.session_id)
    await client.expire(index_key, 2)
    before_session_ttl = await client.ttl(session_key)

    preview = await repair.run(dry_run=True)
    assert preview.ttl_violations == 1
    assert preview.memberships_removed == 0
    assert preview.indexes[0].ttl_after >= 170

    applied = await repair.run(dry_run=False)
    assert applied.memberships_removed == 0
    assert await client.ttl(index_key) >= 170
    assert await client.ttl(session_key) <= before_session_ttl
    again = await repair.run(dry_run=False)
    assert again.memberships_removed == 0
    assert again.ttl_violations == 0


async def test_removes_missing_and_malformed_members_but_not_records(
    repair_context: tuple[Redis, str, RefreshTokenRepository, SessionIndexRepair],
) -> None:
    client, _prefix, store, repair = repair_context
    user_id = _user_id()
    index_key = store._user_sessions_key(user_id)
    await client.sadd(index_key, "missing", "malformed")
    malformed_key = store._session_key("malformed")
    await client.set(malformed_key, "not-json", ex=180)
    await client.expire(index_key, 180)

    report = await repair.run(dry_run=False)
    assert report.memberships_removed == 2
    assert report.malformed_records == 1
    assert await client.exists(index_key) == 0
    assert await client.get(malformed_key) == "not-json"


async def test_repairs_orphan_and_recreates_disappeared_index(
    repair_context: tuple[Redis, str, RefreshTokenRepository, SessionIndexRepair],
) -> None:
    client, _prefix, store, repair = repair_context
    session = _session(_user_id(), ttl=120)
    await client.set(store._session_key(session.session_id), session.model_dump_json(), ex=session.ttl)
    report = await repair.run(dry_run=False, orphan_limit=100)
    assert report.orphan_sessions == 1
    assert report.orphan_memberships_restored == 1
    assert await client.sismember(store._user_sessions_key(session.user_id), session.session_id) == 1
    assert await client.ttl(store._user_sessions_key(session.user_id)) >= 110


async def test_cursor_batch_and_wrong_type_failure_do_not_block_other_indexes(
    repair_context: tuple[Redis, str, RefreshTokenRepository, SessionIndexRepair],
) -> None:
    client, _prefix, store, repair = repair_context
    for index_number in range(9):
        user_id = f"{index_number:024x}"
        session = _session(user_id, ttl=120)
        assert isinstance(await store.store_session(session), Success)
    await client.set(store._user_sessions_key("wrong-type"), "not-a-set")
    repair = SessionIndexRepair(client, key_prefix=store._key_prefix, batch_limit=3, scan_count=1)
    cursor = 0
    scanned = 0
    failures = 0
    for _ in range(20):
        result = await repair.run(cursor=cursor, dry_run=True)
        scanned += result.scanned_indexes
        failures += result.failures
        cursor = result.next_cursor
        if result.complete:
            break
    assert scanned >= 10
    assert failures == 1


async def test_concurrent_replacement_wins_over_stale_repair_observation(
    repair_context: tuple[Redis, str, RefreshTokenRepository, SessionIndexRepair],
) -> None:
    client, _prefix, store, repair = repair_context
    user_id = _user_id()
    session_id = "race-session"
    old_key = store._session_key(session_id)
    index_key = store._user_sessions_key(user_id)
    await client.set(old_key, "not-json", ex=180)
    await client.sadd(index_key, session_id)
    replacement = _session(user_id, ttl=180, session_id=session_id)
    await asyncio.gather(repair.run(dry_run=False), store.store_session(replacement))
    assert await client.exists(old_key) == 1
    assert await client.sismember(index_key, session_id) == 1


async def test_repeated_repair_and_revoke_interleavings_never_resurrect(
    repair_context: tuple[Redis, str, RefreshTokenRepository, SessionIndexRepair],
) -> None:
    client, _prefix, store, repair = repair_context
    for _ in range(12):
        session = _session(_user_id(), ttl=120)
        assert isinstance(await store.store_session(session), Success)
        await client.expire(store._user_sessions_key(session.user_id), 2)
        await asyncio.gather(
            repair.run(dry_run=False),
            store.revoke_session(session.session_id, session.user_id),
        )
        assert await client.exists(store._session_key(session.session_id)) == 0
        assert await client.sismember(store._user_sessions_key(session.user_id), session.session_id) == 0
