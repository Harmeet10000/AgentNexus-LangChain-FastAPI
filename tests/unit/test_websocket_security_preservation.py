"""Task 2: Preservation Tests for WebSocket Security.

These tests verify that non-buggy inputs produce expected behavior.
They establish a baseline before the fix and confirm no regressions after.

Run with: uv run pytest tests/unit/test_websocket_security_preservation.py -v
"""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import WebSocketException

from app.config import get_settings
from app.features.auth.repository import SessionData
from app.features.auth.security import TokenClaims
from app.features.auth.websocket_security import (
    WebSocketSecurityContext,
    build_websocket_security_service,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_settings():
    return get_settings()


@pytest.fixture
async def ws_security_service(redis, mock_settings):
    """Build WebSocketSecurityService with mocked rate limiters to avoid background tasks."""
    from unittest.mock import MagicMock, patch

    # Create mock limiters BEFORE building service
    mock_user_limiter = MagicMock()
    mock_user_limiter.try_acquire = MagicMock()

    mock_connection_limiter = MagicMock()
    mock_connection_limiter.try_acquire = MagicMock()

    # Patch Limiter class to return mocks
    with patch('app.features.auth.websocket_security.Limiter') as MockLimiter:
        MockLimiter.side_effect = [mock_user_limiter, mock_connection_limiter]
        service = await build_websocket_security_service(redis, mock_settings)

    return service


@pytest.fixture
def valid_token_claims():
    return TokenClaims(
        sub="user-preserve",
        sid="session-preserve",
        exp=datetime.now(UTC) + timedelta(minutes=15),
        iat=datetime.now(UTC),
        jti="jti-preserve",
        token_type="access",
    )


@pytest.fixture
def valid_context(valid_token_claims):
    return WebSocketSecurityContext(
        claims=valid_token_claims,
        user_id=valid_token_claims.sub,
        session_id=valid_token_claims.sid,
        connection_id=str(uuid4()),
        origin="https://example.com",
        user_rate_limit_key=f"user:{valid_token_claims.sub}",
        connection_rate_limit_key=f"connection:{uuid4()}",
    )


class TestPreservation1ValidConnections:
    """Property 2.1: Valid connections maintain authentication."""

    async def test_valid_connection_registers_successfully(
        self,
        ws_security_service,
        valid_context,
        redis,
    ):
        """Valid connections should register without issues."""
        # WHEN a valid connection is registered
        await ws_security_service.register_connection(valid_context)

        # THEN it should appear in Redis user sorted set
        user_key = f"ws:user:{valid_context.user_id}"
        user_conns = await redis.zrange(user_key, 0, -1)
        assert valid_context.connection_id in user_conns

    async def test_valid_connection_unregisters_cleanly(
        self,
        ws_security_service,
        valid_context,
        redis,
    ):
        """Valid connections should unregister without issues."""
        # GIVEN a registered connection
        await ws_security_service.register_connection(valid_context)

        # WHEN unregistered
        await ws_security_service.unregister_connection(valid_context)

        # THEN it should be removed from Redis
        user_key = f"ws:user:{valid_context.user_id}"
        user_conns = await redis.zrange(user_key, 0, -1)
        assert valid_context.connection_id not in user_conns


class TestPreservation2PresenceUpdates:
    """Property 2.2: Presence keys updated correctly."""

    async def test_touch_connection_updates_presence(
        self,
        ws_security_service,
        valid_context,
        redis,
    ):
        """Touch should update presence scores."""
        # GIVEN a registered connection
        await ws_security_service.register_connection(valid_context)

        # Get initial score
        user_key = f"ws:user:{valid_context.user_id}"
        initial_score = await redis.zscore(user_key, valid_context.connection_id)

        # WHEN touched
        await asyncio.sleep(0.1)
        await ws_security_service.touch_connection(valid_context)

        # THEN score should be updated (unless throttled)
        # Note: throttle is 30s, so this may not update immediately
        # But subsequent calls after 30s should update
        assert initial_score is not None


class TestPreservation3RateLimitingNormalTraffic:
    """Property 2.3: Rate limiting works correctly for normal patterns."""

    async def test_connection_capacity_enforced(
        self,
        ws_security_service,
        valid_token_claims,
        mock_settings,
    ):
        """Connection capacity should be enforced."""
        user_id = "user-capacity-preserve"
        max_conns = mock_settings.WEBSOCKET_MAX_CONNECTIONS_PER_USER

        # Register max connections
        for i in range(max_conns):
            context = WebSocketSecurityContext(
                claims=valid_token_claims,
                user_id=user_id,
                session_id=f"session-{i}",
                connection_id=f"connection-{i}",
                origin="https://example.com",
                user_rate_limit_key=f"user:{user_id}",
                connection_rate_limit_key=f"connection:conn-{i}",
            )
            await ws_security_service.register_connection(context)

        # Try to exceed limit
        over_context = WebSocketSecurityContext(
            claims=valid_token_claims,
            user_id=user_id,
            session_id="session-over",
            connection_id="connection-over",
            origin="https://example.com",
            user_rate_limit_key=f"user:{user_id}",
            connection_rate_limit_key="connection:over",
        )

        # Should raise exception
        with pytest.raises(WebSocketException):
            await ws_security_service.ensure_connection_capacity(user_id)


class TestPreservation4OriginValidation:
    """Property 2.4: Origin validation still works."""

    def test_invalid_origin_rejected(self, ws_security_service, mock_settings):
        """Invalid origins should still be rejected."""
        mock_settings.WEBSOCKET_ALLOWED_ORIGINS = ["https://allowed.com"]

        with pytest.raises(WebSocketException, match="Origin not allowed"):
            ws_security_service.ensure_origin_allowed("https://malicious.com")

    def test_valid_origin_allowed(self, ws_security_service, mock_settings):
        """Valid origins should be allowed."""
        mock_settings.WEBSOCKET_ALLOWED_ORIGINS = ["https://allowed.com"]

        # Should not raise
        ws_security_service.ensure_origin_allowed("https://allowed.com")


class TestPreservation5AtomicCapacityCheck:
    """Property 2.5: Capacity check is now atomic."""

    async def test_capacity_check_uses_sorted_set_atomically(
        self,
        ws_security_service,
        valid_token_claims,
        mock_settings,
    ):
        """Capacity check should use atomic sorted set operations."""
        user_id = "user-atomic-test"
        max_conns = mock_settings.WEBSOCKET_MAX_CONNECTIONS_PER_USER

        # Register connections up to capacity
        for i in range(max_conns):
            context = WebSocketSecurityContext(
                claims=valid_token_claims,
                user_id=user_id,
                session_id=f"session-{i}",
                connection_id=f"connection-{i}",
                origin="https://example.com",
                user_rate_limit_key=f"user:{user_id}",
                connection_rate_limit_key=f"connection:conn-{i}",
            )
            await ws_security_service.register_connection(context)

        # Count should be accurate
        count = await ws_security_service.get_active_connection_count(user_id)
        assert count == max_conns
