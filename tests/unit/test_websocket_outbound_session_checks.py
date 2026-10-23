"""Outbound WebSocket revocation checks stay bounded off the token hot path."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest
from returns.result import Failure, Success

from app.config import get_settings
from app.features.auth.repository import SessionData
from app.features.auth.websocket_security import (
    WebSocketSecurityContext,
    WebSocketSecurityService,
    WebSocketSessionRevokedError,
)
from app.security.tokens import TokenClaims

if TYPE_CHECKING:
    from typing import Any

    from app.features.auth.repository import RefreshTokenRepository

pytestmark = pytest.mark.unit


class _TokenRepo:
    def __init__(self, *results: Any) -> None:
        self._results = results
        self.calls = 0

    async def get_session(self, _session_id: str) -> Any:
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


class _Limiter:
    def try_acquire(self, _key: str) -> bool:
        return True

    def buckets(self) -> tuple[()]:
        return ()


class _BlockedTokenRepo(_TokenRepo):
    def __init__(self) -> None:
        super().__init__(Success(_valid_session()))
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_session(self, session_id: str) -> Any:
        self.started.set()
        await self.release.wait()
        return await super().get_session(session_id)


def _context() -> WebSocketSecurityContext:
    claims = TokenClaims(
        sub="user-1",
        sid="session-1",
        jti="jti-1",
        role="user",
        permissions=[],
        token_type="access",
    )
    return WebSocketSecurityContext(
        claims=claims,
        user_id=claims.sub,
        session_id=claims.sid,
        connection_id="connection-1",
        origin="https://example.com",
        user_rate_limit_key="user:user-1",
        connection_rate_limit_key="connection:connection-1",
    )


def _valid_session() -> SessionData:
    return SessionData(
        session_id="session-1",
        user_id="user-1",
        device_id="device-1",
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        ttl=3600,
    )


def _service(repo: _TokenRepo) -> WebSocketSecurityService:
    limiter = _Limiter()
    return WebSocketSecurityService(
        redis=None,
        settings=get_settings(),
        user_limiter=limiter,  # type: ignore[arg-type] — narrow limiter fake
        connection_limiter=limiter,  # type: ignore[arg-type]
        token_repo=cast("RefreshTokenRepository", repo),
    )


async def test_outbound_burst_checks_session_once_but_sends_every_frame() -> None:
    repo = _TokenRepo(Success(_valid_session()))
    service = _service(repo)
    websocket = AsyncMock()
    context = _context()
    try:
        for sequence in range(500):
            await service.send_json(websocket, {"sequence": sequence}, context)
    finally:
        service.close()

    assert repo.calls == 1
    assert websocket.send_json.await_count == 500


async def test_due_outbound_check_blocks_revoked_session_before_wire_send() -> None:
    repo = _TokenRepo(Success(_valid_session()), Success(None))
    service = _service(repo)
    websocket = AsyncMock()
    context = _context()
    try:
        await service.send_json(websocket, {"sequence": 1}, context)
        service._last_outbound_session_check[context.connection_id] -= (
            service._settings.WEBSOCKET_OUTBOUND_SESSION_CHECK_INTERVAL_SECONDS + 1
        )
        with pytest.raises(WebSocketSessionRevokedError):
            await service.send_json(websocket, {"sequence": 2}, context)
    finally:
        service.close()

    assert repo.calls == 2
    assert websocket.send_json.await_count == 1


async def test_failed_outbound_check_fails_closed_before_wire_send() -> None:
    repo = _TokenRepo(Failure(RuntimeError("redis unavailable")))
    service = _service(repo)
    websocket = AsyncMock()
    context = _context()
    try:
        with pytest.raises(WebSocketSessionRevokedError):
            await service.send_json(websocket, {"sequence": 1}, context)
    finally:
        service.close()

    assert repo.calls == 1
    assert websocket.send_json.await_count == 0
    assert context.connection_id not in service._last_outbound_session_check


async def test_failed_due_check_does_not_refresh_previous_success_timestamp() -> None:
    repo = _TokenRepo(Success(_valid_session()), Failure(RuntimeError("redis unavailable")))
    service = _service(repo)
    websocket = AsyncMock()
    context = _context()
    try:
        await service.send_json(websocket, {"sequence": 1}, context)
        previous_check = service._last_outbound_session_check[context.connection_id]
        service._last_outbound_session_check[context.connection_id] -= (
            service._settings.WEBSOCKET_OUTBOUND_SESSION_CHECK_INTERVAL_SECONDS + 1
        )
        stale_check = service._last_outbound_session_check[context.connection_id]
        with pytest.raises(WebSocketSessionRevokedError):
            await service.send_json(websocket, {"sequence": 2}, context)
    finally:
        service.close()

    assert repo.calls == 2
    assert websocket.send_json.await_count == 1
    assert service._last_outbound_session_check[context.connection_id] == stale_check
    assert service._last_outbound_session_check[context.connection_id] < previous_check


async def test_concurrent_send_cannot_bypass_in_flight_session_check() -> None:
    repo = _BlockedTokenRepo()
    service = _service(repo)
    websocket = AsyncMock()
    context = _context()
    first = asyncio.create_task(service.send_json(websocket, {"sequence": 1}, context))
    await repo.started.wait()
    second = asyncio.create_task(service.send_json(websocket, {"sequence": 2}, context))
    try:
        await asyncio.sleep(0)
        assert websocket.send_json.await_count == 0
        repo.release.set()
        await asyncio.gather(first, second)
    finally:
        first.cancel()
        second.cancel()
        service.close()

    assert repo.calls == 1
    assert websocket.send_json.await_count == 2


async def test_inbound_frames_still_check_session_every_time() -> None:
    repo = _TokenRepo(Success(_valid_session()))
    service = _service(repo)
    websocket = AsyncMock()
    websocket.receive_json.side_effect = [{"sequence": 1}, {"sequence": 2}]
    context = _context()
    try:
        assert await service.receive_json(websocket, context) == {"sequence": 1}
        assert await service.receive_json(websocket, context) == {"sequence": 2}
    finally:
        service.close()

    assert repo.calls == 2


async def test_unregister_clears_all_connection_local_state_without_redis() -> None:
    repo = _TokenRepo(Success(_valid_session()))
    service = _service(repo)
    context = _context()
    websocket = AsyncMock()
    service._last_touch_time[context.connection_id] = 1
    service._presence_score[context.connection_id] = 1
    service._last_outbound_session_check[context.connection_id] = 1
    service._outbound_session_check_locks[context.connection_id] = asyncio.Lock()
    service._live_connections[context.connection_id] = (websocket, context)
    try:
        await service.unregister_connection(context)
    finally:
        service.close()

    assert context.connection_id not in service._last_touch_time
    assert context.connection_id not in service._presence_score
    assert context.connection_id not in service._last_outbound_session_check
    assert context.connection_id not in service._outbound_session_check_locks
    assert context.connection_id not in service._live_connections
