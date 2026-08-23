import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from returns.result import Success

from app.features.auth.dto import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
)
from app.utils.exceptions import (
    ConflictException,
    NotFoundException,
    UnauthorizedException,
)

# Stale against the Result-pattern service (returns AppResult, not bare values);
# deferred until the mocks are updated. Runs with the integration suite.
pytestmark = pytest.mark.integration


def make_mock_user(**kwargs):
    user = MagicMock()
    user.id = kwargs.get("id", "507f1f77bcf86cd799439011")
    user.email = kwargs.get("email", "test@example.com")
    user.hashed_password = kwargs.get("hashed_password", "$argon2id$v=19$m=65536,t=3,p=4$abc")
    user.is_active = kwargs.get("is_active", True)
    user.is_verified = kwargs.get("is_verified", True)
    role_value = kwargs.get("role", "user")
    user.role = MagicMock()
    user.role.value = role_value
    user.full_name = kwargs.get("full_name")
    user.oauth_accounts = []
    user.verification_token_hash = None
    user.reset_token_hash = None
    user.reset_token_expires_at = None
    user.created_at = datetime.now(UTC)
    user.updated_at = datetime.now(UTC)
    return user


def make_session_json(session_id="sid-1", user_id="507f1f77bcf86cd799439011"):
    return json.dumps(
        {
            "session_id": session_id,
            "user_id": user_id,
            "device_id": "dev-1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "ttl": 3600,
            "device_name": None,
            "ip_address": None,
            "user_agent": None,
        }
    )


class TestAuthRegister:
    async def test_register_creates_user(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_repo.email_exists.return_value = False
        mock_created = make_mock_user(email="new@example.com", is_verified=False)
        mock_repo.create.return_value = mock_created

        with patch("app.features.auth.service.User", return_value=mock_created):
            dto = RegisterRequest(email="new@example.com", password="ValidPass1")
            result = await auth_service.register(dto)

        assert result.email == "new@example.com"
        assert result.is_verified is False
        assert result.is_active is True

    async def test_register_duplicate_email_raises_conflict(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_repo.email_exists.return_value = True

        dto = RegisterRequest(email="existing@example.com", password="ValidPass1")
        with pytest.raises(ConflictException, match="Email already registered"):
            await auth_service.register(dto)

    async def test_register_hashes_password(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_repo.email_exists.return_value = False
        mock_created = make_mock_user(hashed_password="hashed_abc")
        mock_repo.create.return_value = mock_created

        with (
            patch("app.features.auth.service.User", return_value=mock_created),
            patch("app.features.auth.service.hash_password", return_value="hashed_abc"),
        ):
            dto = RegisterRequest(email="test@example.com", password="ValidPass1")
            await auth_service.register(dto)

            created_user = mock_repo.create.call_args[0][0]
            assert created_user.hashed_password == "hashed_abc"
            assert created_user.hashed_password != "ValidPass1"


class TestAuthLogin:
    async def test_login_valid_credentials(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_user = make_mock_user(is_verified=True)
        mock_repo.find_by_email.return_value = mock_user

        with (
            patch("app.features.auth.service.verify_password", return_value=True),
            patch("app.features.auth.service.needs_rehash", return_value=False),
            patch.object(
                auth_service,
                "_create_session",
                return_value=TokenResponse(
                    access_token="mock-at",
                    refresh_token="mock-rt",
                    token_type="bearer",
                    expires_in=3600,
                ),
            ),
        ):
            result = await auth_service.login(
                LoginRequest(email="test@example.com", password="ValidPass1")
            )

            assert result.access_token == "mock-at"
            assert result.refresh_token == "mock-rt"
            assert result.token_type == "bearer"

    async def test_login_wrong_password_raises_unauthorized(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_user = make_mock_user(is_verified=True)
        mock_repo.find_by_email.return_value = mock_user

        with (
            patch("app.features.auth.service.verify_password", return_value=False),
            pytest.raises(UnauthorizedException, match="Invalid credentials"),
        ):
            await auth_service.login(LoginRequest(email="test@example.com", password="WrongPass1"))

    async def test_login_nonexistent_email_raises_unauthorized(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_repo.find_by_email.return_value = None

        with pytest.raises(UnauthorizedException, match="Invalid credentials"):
            await auth_service.login(
                LoginRequest(email="nonexistent@example.com", password="AnyPass1")
            )

    async def test_login_disabled_account_raises_unauthorized(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_user = make_mock_user(is_active=False, is_verified=True)
        mock_repo.find_by_email.return_value = mock_user

        with (
            patch("app.features.auth.service.verify_password", return_value=True),
            pytest.raises(UnauthorizedException, match="Account is disabled"),
        ):
            await auth_service.login(
                LoginRequest(email="disabled@example.com", password="ValidPass1")
            )

    async def test_login_unverified_email_raises_unauthorized(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_user = make_mock_user(is_verified=False)
        mock_repo.find_by_email.return_value = mock_user

        with (
            patch("app.features.auth.service.verify_password", return_value=True),
            pytest.raises(UnauthorizedException, match="Email not verified"),
        ):
            await auth_service.login(
                LoginRequest(email="unverified@example.com", password="ValidPass1")
            )


class TestAuthRefresh:
    async def test_refresh_valid_token(self, auth_service, redis):
        mock_repo = auth_service._user_repo
        mock_user = make_mock_user()
        mock_repo.find_by_id_result.return_value = Success(mock_user)

        session_id = "test-session-id"
        await redis.set(
            f"auth:session:{session_id}",
            make_session_json(session_id=session_id, user_id=mock_user.id),
        )

        with (
            patch(
                "app.features.auth.service.decode_token",
                return_value=MagicMock(token_type="refresh", jti=session_id, sub=mock_user.id),
            ),
            patch("app.features.auth.service.create_access_token", return_value=("mock-at", 3600)),
        ):
            result = await auth_service.refresh(session_id)

            assert result.access_token == "mock-at"
            assert result.refresh_token == session_id

    async def test_refresh_expired_token_raises_unauthorized(self, auth_service):
        with (
            patch(
                "app.features.auth.service.decode_token",
                return_value=MagicMock(token_type="refresh", jti="nonexistent", sub="user-id"),
            ),
            pytest.raises(UnauthorizedException),
        ):
            await auth_service.refresh("nonexistent-session")

    async def test_refresh_revoked_token_raises_unauthorized(self, auth_service):
        with (
            patch(
                "app.features.auth.service.decode_token",
                return_value=MagicMock(token_type="refresh", jti="unknown", sub="user-id"),
            ),
            pytest.raises(UnauthorizedException),
        ):
            await auth_service.refresh("unknown-token")


class TestAuthLogout:
    async def test_logout_revokes_token(self, auth_service, redis):
        mock_repo = auth_service._user_repo
        mock_repo.find_by_id.return_value = make_mock_user()

        session_id = "test-session-id"
        await redis.set(f"auth:session:{session_id}", make_session_json(session_id=session_id))

        with patch(
            "app.features.auth.service.decode_token",
            return_value=MagicMock(token_type="refresh", jti=session_id, sub="user-id"),
        ):
            await auth_service.logout(session_id)

            cached = await redis.get(f"auth:session:{session_id}")
            assert cached is None


class TestAuthSessions:
    async def test_list_sessions(self, auth_service, redis):
        session_id = "active-session-1"
        user_id = "user-123"
        await redis.sadd(f"auth:user_sessions:{user_id}", session_id)
        await redis.set(
            f"auth:session:{session_id}", make_session_json(session_id=session_id, user_id=user_id)
        )

        sessions = await auth_service.list_sessions(user_id)
        assert isinstance(sessions, list)

    async def test_revoke_session(self, auth_service, redis):
        session_id = "revoke-me"
        user_id = "user-123"
        await redis.sadd(f"auth:user_sessions:{user_id}", session_id)
        await redis.set(
            f"auth:session:{session_id}", make_session_json(session_id=session_id, user_id=user_id)
        )

        await auth_service.revoke_session(session_id, user_id)

        cached = await redis.get(f"auth:session:{session_id}")
        assert cached is None

    async def test_revoke_all_sessions_except_current(self, auth_service, redis):
        user_id = "user-123"
        keep_session = "keep-me"
        remove_session = "remove-me"
        await redis.sadd(f"auth:user_sessions:{user_id}", keep_session)
        await redis.sadd(f"auth:user_sessions:{user_id}", remove_session)
        await redis.set(
            f"auth:session:{keep_session}",
            make_session_json(session_id=keep_session, user_id=user_id),
        )
        await redis.set(
            f"auth:session:{remove_session}",
            make_session_json(session_id=remove_session, user_id=user_id),
        )

        await auth_service.revoke_all_sessions(user_id, except_session_id=keep_session)

        kept = await redis.get(f"auth:session:{keep_session}")
        removed = await redis.get(f"auth:session:{remove_session}")
        assert kept is not None
        assert removed is None


class TestAuthPasswordReset:
    async def test_forgot_password_generates_token(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_user = make_mock_user(is_verified=True)
        mock_repo.find_by_email.return_value = mock_user

        with (
            patch("app.features.auth.service.datetime") as mock_dt,
        ):
            mock_dt.timezone = MagicMock()
            mock_dt.timezone.utc = UTC
            mock_dt.now.return_value = datetime(2026, 1, 1, tzinfo=UTC)
            mock_dt.timedelta = timedelta

            await auth_service.forgot_password("test@example.com")
            assert mock_user.reset_token_hash is not None

    async def test_reset_password_updates_password(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_user = make_mock_user(is_verified=True)
        mock_user.reset_token_expires_at = datetime(2099, 1, 1, tzinfo=UTC)
        mock_repo.find_by_reset_token_hash.return_value = mock_user

        with patch("app.features.auth.service.hash_password", return_value="new_hashed_password"):
            await auth_service.reset_password("valid-token", "NewPass123")
            assert mock_user.hashed_password == "new_hashed_password"

    async def test_reset_password_invalid_token_raises(self, auth_service):
        mock_repo = auth_service._user_repo
        mock_repo.find_by_reset_token_hash.return_value = None

        with pytest.raises(NotFoundException, match="Invalid or expired"):
            await auth_service.reset_password("invalid-token", "NewPass123")
