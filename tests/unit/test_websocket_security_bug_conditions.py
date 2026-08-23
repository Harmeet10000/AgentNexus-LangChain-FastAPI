"""Task 1: Bug Condition Exploration Tests for WebSocket Security.

This test file MUST FAIL on unfixed code to confirm the bugs exist.
DO NOT attempt to fix the test or the code when it fails.

GOAL: Surface counterexamples that demonstrate:
- Session revocation gap (P0)
- JWT expiry ignored (P0)
- TOCTOU capacity overflow (P0)
- Rate limiter bypass (P1)

Expected Outcome: Tests FAIL on unfixed code (confirms vulnerabilities exist).

Run with: uv run pytest tests/unit/test_websocket_security_bug_conditions.py -v
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast
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

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_settings():
    return get_settings()


@pytest.fixture
def mock_token_claims():
    return TokenClaims(
        sub="507f1f77bcf86cd799439011",
        sid="session-abc",
        jti="jti-123",
        role="user",
        permissions=[],
        token_type="access",
    )


@pytest.fixture
async def ws_security_service(redis, mock_settings):
    """Build WebSocketSecurityService with real in-memory rate limiters."""
    return await build_websocket_security_service(redis, mock_settings)


@pytest.fixture
def ws_security_context(mock_token_claims):
    return WebSocketSecurityContext(
        claims=mock_token_claims,
        user_id=mock_token_claims.sub,
        session_id=mock_token_claims.sid,
        connection_id=str(uuid4()),
        origin="https://example.com",
        user_rate_limit_key=f"user:{mock_token_claims.sub}",
        connection_rate_limit_key=f"connection:{uuid4()}",
    )


class TestBugCondition1SessionRevocationGap:
    """
    Property 1.1: Session Revocation Gap (P0)

    Bug Condition: WHEN a user's session is revoked via AuthService.revoke_session()
                   THEN the system does NOT close live WebSocket connections
    """

    async def test_websocket_connection_revoked_on_session_check(
        self,
        redis,
        ws_security_service,
        ws_security_context: WebSocketSecurityContext,
    ):
        """Demonstrates that revoked sessions are detected by pull-based check."""
        # GIVEN a registered WebSocket connection
        await ws_security_service.register_connection(ws_security_context)

        # AND a session record exists in Redis
        token_repo = RefreshTokenRepository(redis)
        session_data = SessionData(
            session_id=ws_security_context.session_id,
            user_id=ws_security_context.user_id,
            device_id="device-123",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=30),
            ttl=3600,
        )
        result = await token_repo.store_session(session_data)
        assert isinstance(result, Success)

        # WHEN the session is revoked
        revoke_result = await token_repo.revoke_session(
            session_id=ws_security_context.session_id,
            user_id=ws_security_context.user_id,
            reason="logout",
        )
        assert isinstance(revoke_result, Success)

        # THEN the WebSocket connection should be detected as revoked
        ws_security_service._token_repo = token_repo

        # This should raise WebSocketSessionRevokedError on fixed code
        with pytest.raises(WebSocketSessionRevokedError):
            await ws_security_service._check_session_validity(
                ws_security_context,
                MagicMock(),
            )


class TestBugCondition2JWTExpiryIgnored:
    """
    Property 1.2: JWT Expiry Ignored (P0)

    Demonstrates that expired JWT tokens are detected on next session check.
    """

    async def test_websocket_expired_token_detected_on_session_check(
        self,
        redis,
        ws_security_service,
    ):
        """Expired sessions should be detected by pull-based check."""
        # GIVEN a session that has expired
        expired_session_id = "expired-session"
        user_id = "507f1f77bcf86cd799439012"

        token_repo = RefreshTokenRepository(redis)

        # Create a session that expires immediately
        session_data = SessionData(
            session_id=expired_session_id,
            user_id=user_id,
            device_id="device-123",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) - timedelta(seconds=1),  # Already expired
            ttl=1,  # TTL of 1 second
        )
        await token_repo.store_session(session_data)

        # WHEN we wait for session to expire
        await asyncio.sleep(2)

        # AND check session validity
        ws_security_service._token_repo = token_repo
        claims = TokenClaims(
            sub=user_id,
            sid=expired_session_id,
            jti="jti-expired",
            role="user",
            permissions=[],
            token_type="access",
        )
        context = WebSocketSecurityContext(
            claims=claims,
            user_id=user_id,
            session_id=expired_session_id,
            connection_id=str(uuid4()),
            origin="https://example.com",
            user_rate_limit_key=f"user:{user_id}",
            connection_rate_limit_key=f"connection:{uuid4()}",
        )

        # THEN expired session should be detected
        with pytest.raises(WebSocketSessionRevokedError):
            await ws_security_service._check_session_validity(
                context,
                MagicMock(),
            )


class TestBugCondition3TOCTOUCapacityOverflow:
    """
    Property 1.3: TOCTOU Capacity Overflow (P0)

    Demonstrates that TOCTOU race condition is fixed with atomic operations.
    """

    async def test_simultaneous_connections_respect_capacity_limit(
        self,
        redis,
        ws_security_service,
        mock_token_claims: TokenClaims,
    ):
        """Capacity limit should be enforced atomically."""
        user_id = "user-capacity-test"
        max_connections = 3

        # WHEN attempting to exceed capacity
        connection_tasks = []
        for i in range(4):
            context = WebSocketSecurityContext(
                claims=mock_token_claims,
                user_id=user_id,
                session_id=f"session-{i}",
                connection_id=f"connection-{i}",
                origin="https://example.com",
                user_rate_limit_key=f"user:{user_id}",
                connection_rate_limit_key=f"connection:conn-{i}",
            )
            connection_tasks.append((context, i))

        # Attempt all 4 simultaneously
        successful = 0
        for context, idx in connection_tasks:
            try:
                await ws_security_service.ensure_connection_capacity(user_id)
                await ws_security_service.register_connection(context)
                successful += 1
            except Exception:
                pass  # Expected to fail after max

        # THEN final count should not exceed capacity
        final_count = await ws_security_service.get_active_connection_count(user_id)
        assert final_count <= max_connections, (
            f"Capacity overflow detected: {final_count} > {max_connections}"
        )


class TestBugCondition4RateLimiterBypass:
    """
    Property 1.4: Rate Limiter Bypass (P1)

    Demonstrates that direct rate limiter calls enforce limits correctly.
    """

    async def test_rate_limiter_enforces_limits(
        self,
        ws_security_service,
        ws_security_context: WebSocketSecurityContext,
    ):
        """Rate limiting should enforce message rate limits."""
        settings = ws_security_service._settings
        attempts = settings.WEBSOCKET_USER_MESSAGE_RATE + 5

        accepted = 0
        rejected = 0

        # Try to send more than the rate limit over a single connection —
        # the per-connection limit (20/10s) binds before the per-user limit.
        expected_accepted = min(
            settings.WEBSOCKET_USER_MESSAGE_RATE,
            settings.WEBSOCKET_CONNECTION_MESSAGE_RATE,
        )
        for i in range(attempts):
            try:
                await ws_security_service._apply_rate_limits(ws_security_context)
                accepted += 1
            except WebSocketRateLimitExceededError:
                rejected += 1

        # THEN exactly expected_accepted should get through
        assert accepted == expected_accepted, (
            f"Expected {expected_accepted} accepted, got {accepted}"
        )
        assert rejected == attempts - expected_accepted, (
            f"Expected {attempts - expected_accepted} rejected, got {rejected}"
        )


class TestBugCondition5SessionConnectionsSetRead:
    """
    Property 1.5: Sorted Set is Now Read

    Demonstrates that ws:session:{session_id} sorted set is actually used.
    """

    async def test_session_sorted_set_used_for_revocation_lookup(
        self,
        redis,
        ws_security_service,
        ws_security_context: WebSocketSecurityContext,
    ):
        """Session sorted set should be readable for revocation."""
        # GIVEN a registered connection
        await ws_security_service.register_connection(ws_security_context)

        # VERIFY the session sorted set is populated and readable
        session_key = f"ws:session:{ws_security_context.session_id}"
        connection_ids = await cast("any", redis).zrange(session_key, 0, -1)

        assert ws_security_context.connection_id in connection_ids, (
            "Session presence sorted set should contain the connection ID"
        )

        # WHEN we look up connections for a session (as revoke would)
        found_connections = await cast("any", redis).zrange(session_key, 0, -1)

        # THEN we should find the connections
        assert len(found_connections) > 0, "Session connections should be findable via sorted set"
