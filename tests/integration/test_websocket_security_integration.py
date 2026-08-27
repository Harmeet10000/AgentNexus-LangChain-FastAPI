"""Integration Tests for WebSocket Security Bugfix.

Tests the full flow of:
1. Session revocation reaching open sockets
2. Rate limiting enforcement
3. Capacity enforcement
4. Presence tracking with sorted sets

Run with: uv run pytest tests/integration/test_websocket_security_integration.py -v
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from returns.result import Success

from app.config import get_settings
from app.features.auth.repository import RefreshTokenRepository, SessionData
from app.features.auth.security import TokenClaims
from app.features.auth.websocket_security import (
    WebSocketRateLimitExceededError,
    WebSocketSecurityContext,
    WebSocketSessionRevokedError,
    build_websocket_security_service,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_settings():
    return get_settings()


@pytest.fixture
async def ws_security_with_repo(redis, mock_settings):
    """Build WebSocketSecurityService with token repo and real in-memory rate limiters."""
    service = await build_websocket_security_service(redis, mock_settings)
    service._token_repo = RefreshTokenRepository(redis)
    return service


class TestSessionRevocationIntegration:
    """Full flow: Create session → Open connection → Revoke session → Connection closes"""

    async def test_session_revocation_closes_websocket_connection(
        self,
        redis,
        ws_security_with_repo,
    ):
        """End-to-end: revocation should cause connection to close on next check."""
        # SETUP: Create session and connection
        user_id = "507f1f77bcf86cd799439013"
        session_id = "session-integration-1"
        connection_id = str(uuid4())

        claims = TokenClaims(
            sub=user_id,
            sid=session_id,
            jti="jti-1",
            role="user",
            permissions=[],
            token_type="access",
        )

        context = WebSocketSecurityContext(
            claims=claims,
            user_id=user_id,
            session_id=session_id,
            connection_id=connection_id,
            origin="https://example.com",
            user_rate_limit_key=f"user:{user_id}",
            connection_rate_limit_key=f"connection:{connection_id}",
        )

        # Create and store session
        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            device_id="device-1",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
            ttl=3600,
        )
        result = await ws_security_with_repo._token_repo.store_session(session_data)
        assert isinstance(result, Success)

        # Register connection
        await ws_security_with_repo.register_connection(context)

        # Verify connection is registered
        user_key = f"ws:user:{user_id}"
        conns = await redis.zrange(user_key, 0, -1)
        assert connection_id in conns

        # TRIGGER: Revoke the session
        revoke_result = await ws_security_with_repo._token_repo.revoke_session(
            session_id=session_id,
            user_id=user_id,
            reason="security",
        )
        assert isinstance(revoke_result, Success)

        # VERIFY: Session is gone from Redis
        get_result = await ws_security_with_repo._token_repo.get_session(session_id)
        assert isinstance(get_result, Success)
        assert get_result.unwrap() is None

        # CHECK: Validity check should raise WebSocketSessionRevokedError
        with pytest.raises(WebSocketSessionRevokedError):
            await ws_security_with_repo._check_session_validity(
                context,
                MagicMock(),
            )


class TestRateLimitingIntegration:
    """Full flow: Send messages → Hit rate limit → Messages rejected"""

    async def test_rate_limiting_rejects_messages_over_limit(
        self,
        ws_security_with_repo,
    ):
        """Rate limiter should reject messages exceeding the limit."""
        user_id = "user-ratelimit"
        connection_id = str(uuid4())

        claims = TokenClaims(
            sub=user_id,
            sid="session-rl",
            jti="jti-rl",
            role="user",
            permissions=[],
            token_type="access",
        )

        context = WebSocketSecurityContext(
            claims=claims,
            user_id=user_id,
            session_id="session-rl",
            connection_id=connection_id,
            origin="https://example.com",
            user_rate_limit_key=f"user:{user_id}",
            connection_rate_limit_key=f"connection:{connection_id}",
        )

        # Try to send more messages than the rate limit allows over a single
        # connection — the per-connection limit binds before the per-user limit.
        settings = ws_security_with_repo._settings
        expected_accepted = min(
            settings.WEBSOCKET_USER_MESSAGE_RATE,
            settings.WEBSOCKET_CONNECTION_MESSAGE_RATE,
        )
        attempts = settings.WEBSOCKET_USER_MESSAGE_RATE + 10

        accepted = 0
        rejected = 0

        for _ in range(attempts):
            try:
                await ws_security_with_repo._apply_rate_limits(context)
                accepted += 1
            except WebSocketRateLimitExceededError:
                rejected += 1

        # Verify rate limit was enforced
        assert accepted == expected_accepted, (
            f"Expected {expected_accepted} messages accepted, got {accepted}"
        )
        assert rejected == attempts - expected_accepted, (
            f"Expected {attempts - expected_accepted} messages rejected, got {rejected}"
        )


class TestCapacityEnforcementIntegration:
    """Full flow: Connect → At capacity → Next connection rejected"""

    async def test_capacity_limit_enforced_across_connections(
        self,
        ws_security_with_repo,
    ):
        """Capacity limit should be enforced when connections accumulate."""
        user_id = "user-capacity-integration"
        max_conns = ws_security_with_repo._settings.WEBSOCKET_MAX_CONNECTIONS_PER_USER

        claims = TokenClaims(
            sub=user_id,
            sid="session-cap",
            jti="jti-cap",
            role="user",
            permissions=[],
            token_type="access",
        )

        # Fill up to capacity
        for i in range(max_conns):
            context = WebSocketSecurityContext(
                claims=claims,
                user_id=user_id,
                session_id=f"session-{i}",
                connection_id=f"connection-{i}",
                origin="https://example.com",
                user_rate_limit_key=f"user:{user_id}",
                connection_rate_limit_key=f"connection:{i}",
            )
            await ws_security_with_repo.register_connection(context)

        # Verify we're at capacity
        count = await ws_security_with_repo.get_active_connection_count(user_id)
        assert count == max_conns

        # Try to add one more - should fail
        from fastapi import WebSocketException

        with pytest.raises(WebSocketException, match="Maximum concurrent"):
            await ws_security_with_repo.ensure_connection_capacity(user_id)


class TestPresenceTrackingWithSortedSets:
    """Verify sorted set operations work correctly for presence tracking."""

    async def test_sorted_set_operations_maintain_consistency(
        self,
        redis,
        ws_security_with_repo,
    ):
        """Sorted set operations should maintain consistent state."""
        user_id = "user-presence"
        session_id = "session-presence"

        claims = TokenClaims(
            sub=user_id,
            sid=session_id,
            jti="jti-presence",
            role="user",
            permissions=[],
            token_type="access",
        )

        # Create multiple connections
        connections = []
        for i in range(3):
            context = WebSocketSecurityContext(
                claims=claims,
                user_id=user_id,
                session_id=session_id,
                connection_id=f"conn-{i}",
                origin="https://example.com",
                user_rate_limit_key=f"user:{user_id}",
                connection_rate_limit_key=f"connection:{i}",
            )
            connections.append(context)
            await ws_security_with_repo.register_connection(context)

        # Verify all connections are in user sorted set
        user_key = f"ws:user:{user_id}"
        user_conns = await redis.zrange(user_key, 0, -1)
        assert len(user_conns) == 3

        # Verify all connections are in session sorted set
        session_key = f"ws:session:{session_id}"
        session_conns = await redis.zrange(session_key, 0, -1)
        assert len(session_conns) == 3

        # Verify count operation returns correct value
        count = await ws_security_with_repo.get_active_connection_count(user_id)
        assert count == 3

        # Unregister one connection
        await ws_security_with_repo.unregister_connection(connections[0])

        # Verify it's removed from both sets
        user_conns = await redis.zrange(user_key, 0, -1)
        assert len(user_conns) == 2
        assert "conn-0" not in user_conns

        session_conns = await redis.zrange(session_key, 0, -1)
        assert len(session_conns) == 2
        assert "conn-0" not in session_conns


class TestThrottledTouchConnection:
    """Verify touch_connection throttles Redis writes to reduce hot-path latency."""

    async def test_touch_connection_throttles_redis_writes(
        self,
        redis,
        ws_security_with_repo,
    ):
        """Throttled touch leaves zscore unchanged; post-window touch advances it."""
        from time import time

        user_id = "user-throttle"
        connection_id = str(uuid4())

        claims = TokenClaims(
            sub=user_id,
            sid="session-throttle",
            jti="jti-throttle",
            role="user",
            permissions=[],
            token_type="access",
        )

        context = WebSocketSecurityContext(
            claims=claims,
            user_id=user_id,
            session_id="session-throttle",
            connection_id=connection_id,
            origin="https://example.com",
            user_rate_limit_key=f"user:{user_id}",
            connection_rate_limit_key=f"connection:{connection_id}",
        )

        await ws_security_with_repo.register_connection(context)
        user_key = f"ws:user:{user_id}"
        s0 = await redis.zscore(user_key, connection_id)
        assert s0 is not None

        # Two immediate touches are throttled → score unchanged
        await ws_security_with_repo.touch_connection(context)
        await ws_security_with_repo.touch_connection(context)
        assert await redis.zscore(user_key, connection_id) == s0

        # After throttle window → score advances
        ws_security_with_repo._last_touch_time[connection_id] = time() - 31
        await ws_security_with_repo.touch_connection(context)
        s1 = await redis.zscore(user_key, connection_id)
        assert s1 is not None
        assert float(s1) > float(s0)
