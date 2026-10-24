"""Integration tests for AuthService using InMemoryAuthAccountStore.

These tests verify authentication behavior through the AuthAccountStore port
using the in-memory adapter — no MongoDB required.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fakeredis.aioredis import FakeRedis
from returns.result import Failure

from app.features.auth.account_store import (
    InMemoryAuthAccountStore,
    NewAuthAccount,
    OAuthAccountInput,
)
from app.features.auth.dto import (
    LoginRequest,
    RegisterRequest,
)
from app.features.auth.exceptions import AuthConflictError
from app.features.auth.repository import RefreshTokenRepository
from app.features.auth.service import AuthService
from app.security.passwords import hash_password
from app.security.tokens import decode_token
from app.utils.exceptions import (
    UnauthorizedException,
)

# ─── Test Helpers ───────────────────────────────────────────────────────────────


def make_session_json(session_id: str = "sid-1", user_id: str = "user-123") -> str:
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


async def create_account_in_store(
    account_store: InMemoryAuthAccountStore,
    email: str,
    password: str = "ValidPass1",  # noqa: S107
    is_verified: bool = True,  # Default to verified for tests
    **kwargs,
):
    """Helper to create an account in the store with hashed password."""
    from app.features.auth.account_store import NewAuthAccount
    from app.security.passwords import hash_password

    new_account = NewAuthAccount(
        email=email.lower(),
        password_hash=hash_password(password),
        full_name=kwargs.get("full_name"),
        role=kwargs.get("role", "user"),
        is_verified=is_verified,
        is_active=kwargs.get("is_active", True),
        verification_token_hash=kwargs.get("verification_token_hash"),
    )
    result = await account_store.create(new_account)
    return result.unwrap()


def make_account(
    *,
    id: str = "user-123",
    email: str = "test@example.com",
    hashed_password: str = "$argon2id$v=19$m=65536,t=3,p=4$abc",  # noqa: S107
    is_active: bool = True,
    is_verified: bool = True,
    role: str = "user",
    full_name: str | None = None,
    verification_token_hash: str | None = None,
    reset_token_hash: str | None = None,
    reset_token_expires_at: datetime | None = None,
) -> dict[str, object]:
    """Create an account dict matching AuthAccount structure for in-memory store."""
    now = datetime.now(UTC)
    return {
        "id": id,
        "email": email,
        "password_hash": hashed_password,
        "is_active": is_active,
        "is_verified": is_verified,
        "role": role,
        "permissions": frozenset({"content:read", "profile:read", "profile:write"}),
        "verification_token_hash": verification_token_hash,
        "reset_token_hash": reset_token_hash,
        "reset_token_expires_at": reset_token_expires_at,
        "created_at": now,
        "updated_at": now,
    }


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def redis() -> FakeRedis:
    return FakeRedis(decode_responses=True)


@pytest.fixture
def account_store() -> InMemoryAuthAccountStore:
    return InMemoryAuthAccountStore()


@pytest.fixture
def refresh_token_repo(redis) -> RefreshTokenRepository:
    return RefreshTokenRepository(redis)


@pytest.fixture
def auth_service(account_store, refresh_token_repo) -> AuthService:
    return AuthService(account_store, refresh_token_repo)


# ─── Test Classes ───────────────────────────────────────────────────────────────


class TestAuthRegister:
    """Registration tests."""

    async def test_register_creates_user(self, auth_service: AuthService):
        dto = RegisterRequest(email="new@example.com", password="ValidPass1")
        result = await auth_service.register(dto)
        user = result.unwrap()

        assert user.email == "new@example.com"
        assert user.is_verified is False
        assert user.is_active is True

    async def test_register_duplicate_email_raises_conflict(self, auth_service: AuthService):
        # Pre-create the user
        await create_account_in_store(
            auth_service._account_store, email="existing@example.com", password="ValidPass1"
        )

        dto = RegisterRequest(email="existing@example.com", password="ValidPass1")
        result = await auth_service.register(dto)
        assert isinstance(result, Failure)
        error = result.failure()
        assert error.code == "USER_CONFLICT"
        assert "Email already registered" in error.message

    async def test_register_hashes_password(self, auth_service: AuthService):
        dto = RegisterRequest(email="test@example.com", password="ValidPass1")
        result = await auth_service.register(dto)
        user = result.unwrap()

        assert user.email == "test@example.com"

        # Verify the password was hashed by checking the stored account
        account = await auth_service._account_store.find_by_email("test@example.com")
        account = account.unwrap()
        assert account is not None
        assert account.password_hash != "ValidPass1"
        assert account.password_hash is not None
        assert account.password_hash.startswith("$argon2id$")


class TestAuthLogin:
    """Login tests."""

    async def test_login_valid_credentials(self, auth_service: AuthService):
        # Create a user first
        await create_account_in_store(
            auth_service._account_store, email="test@example.com", password="ValidPass1"
        )

        result = await auth_service.login(
            LoginRequest(email="test@example.com", password="ValidPass1")
        )
        tokens = result.unwrap()

        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert tokens.token_type == "bearer"
        assert tokens.expires_in > 0

    async def test_login_wrong_password_raises_unauthorized(self, auth_service: AuthService):
        # Create a user first
        await create_account_in_store(
            auth_service._account_store, email="test@example.com", password="ValidPass1"
        )

        result = await auth_service.login(
            LoginRequest(email="test@example.com", password="WrongPass1")
        )
        assert isinstance(result, Failure)
        error = result.failure()
        assert "Invalid credentials" in error.message

    async def test_login_nonexistent_email_raises_unauthorized(self, auth_service: AuthService):
        result = await auth_service.login(
            LoginRequest(email="nonexistent@example.com", password="AnyPass1")
        )
        assert isinstance(result, Failure)
        error = result.failure()
        assert "Invalid credentials" in error.message

    async def test_login_unverified_email_raises_unauthorized(self, auth_service: AuthService):
        # Create user without verifying
        await create_account_in_store(
            auth_service._account_store,
            email="unverified@example.com",
            password="ValidPass1",
            is_verified=False,  # Explicitly create unverified user
        )

        result = await auth_service.login(
            LoginRequest(email="unverified@example.com", password="ValidPass1")
        )
        assert isinstance(result, Failure)
        error = result.failure()
        assert "Email not verified" in error.message


class TestAuthRefresh:
    """Token refresh tests."""

    async def test_refresh_valid_token(self, auth_service: AuthService, redis: FakeRedis):
        # Register and login to get a valid session
        await create_account_in_store(
            auth_service._account_store, email="test@example.com", password="ValidPass1"
        )
        login_result = await auth_service.login(
            LoginRequest(email="test@example.com", password="ValidPass1")
        )
        tokens = login_result.unwrap()

        # Use the refresh token from login
        session_result = await auth_service.refresh(tokens.refresh_token)
        session = session_result.unwrap()

        assert session.access_token is not None
        assert session.refresh_token != tokens.refresh_token
        original_claims = decode_token(tokens.refresh_token)
        rotated_claims = decode_token(session.refresh_token)
        assert rotated_claims.family_id == original_claims.family_id
        assert rotated_claims.refresh_generation == 1
        assert rotated_claims.parent_jti == original_claims.jti
        second_result = await auth_service.refresh(session.refresh_token)
        second_claims = decode_token(second_result.unwrap().refresh_token)
        assert second_claims.family_id == original_claims.family_id
        assert second_claims.refresh_generation == 2
        assert second_claims.parent_jti == rotated_claims.jti
        assert session.token_type == "bearer"
        assert session.expires_in > 0

    async def test_refresh_expired_token_raises_unauthorized(self, auth_service: AuthService):
        with pytest.raises(UnauthorizedException, match="Invalid token"):
            await auth_service.refresh("nonexistent-session")

    async def test_refresh_revoked_token_raises_unauthorized(self, auth_service: AuthService):
        with pytest.raises(UnauthorizedException, match="Invalid token"):
            await auth_service.refresh("unknown-token")


class TestAuthLogout:
    """Logout tests."""

    async def test_logout_revokes_token(self, auth_service: AuthService, redis: FakeRedis):
        # Register and login
        await create_account_in_store(
            auth_service._account_store, email="test@example.com", password="ValidPass1"
        )
        login_result = await auth_service.login(
            LoginRequest(email="test@example.com", password="ValidPass1")
        )
        tokens = login_result.unwrap()

        # Logout using the refresh token
        await auth_service.logout(tokens.refresh_token)

        # Verify session is revoked in Redis
        session_id = tokens.refresh_token  # refresh_token is the session_id
        cached = await redis.get(f"auth:session:{session_id}")
        assert cached is None


class TestAuthSessions:
    """Session management tests."""

    async def test_list_sessions(self, auth_service: AuthService, redis: FakeRedis):
        # Register and login to create a session
        await create_account_in_store(
            auth_service._account_store, email="test@example.com", password="ValidPass1"
        )
        login_result = await auth_service.login(
            LoginRequest(email="test@example.com", password="ValidPass1")
        )
        tokens = login_result.unwrap()  # noqa: F841

        # Get the user ID from the account
        account = await auth_service._account_store.find_by_email("test@example.com")
        account = account.unwrap()

        sessions_result = await auth_service.list_sessions(account.id)
        sessions = sessions_result.unwrap()
        assert isinstance(sessions, list)
        assert len(sessions) >= 1

    async def test_revoke_session(self, auth_service: AuthService, redis: FakeRedis):
        await create_account_in_store(
            auth_service._account_store, email="test@example.com", password="ValidPass1"
        )
        login_result = await auth_service.login(
            LoginRequest(email="test@example.com", password="ValidPass1")
        )
        tokens = login_result.unwrap()

        account = await auth_service._account_store.find_by_email("test@example.com")
        account = account.unwrap()

        await auth_service.revoke_session(tokens.refresh_token, account.id)

        cached = await redis.get(f"auth:session:{tokens.refresh_token}")
        assert cached is None

    async def test_revoke_all_sessions_except_current(
        self, auth_service: AuthService, redis: FakeRedis
    ):
        # Create two sessions
        await create_account_in_store(
            auth_service._account_store, email="test@example.com", password="ValidPass1"
        )
        login1 = await auth_service.login(
            LoginRequest(email="test@example.com", password="ValidPass1")
        )
        tokens1 = login1.unwrap()
        login2 = await auth_service.login(
            LoginRequest(email="test@example.com", password="ValidPass1")
        )
        tokens2 = login2.unwrap()

        account = await auth_service._account_store.find_by_email("test@example.com")
        account = account.unwrap()

        # Decode refresh tokens to get session IDs (jti claim)
        session_id1 = decode_token(tokens1.refresh_token).sid
        session_id2 = decode_token(tokens2.refresh_token).sid

        # Revoke all except first session
        await auth_service.revoke_all_sessions(account.id, except_session_id=session_id1)

        kept = await redis.get(f"auth:session:{session_id1}")
        removed = await redis.get(f"auth:session:{session_id2}")
        assert kept is not None
        assert removed is None


class TestAuthPasswordReset:
    """Password reset tests."""

    async def test_forgot_password_generates_token(self, auth_service: AuthService):
        await create_account_in_store(
            auth_service._account_store, email="test@example.com", password="ValidPass1"
        )

        await auth_service.forgot_password("test@example.com")

        # Verify reset token was set
        account = await auth_service._account_store.find_by_email("test@example.com")
        account = account.unwrap()
        assert account.reset_token_hash is not None
        assert account.reset_token_expires_at is not None

    async def test_reset_password_invalid_token_raises(self, auth_service: AuthService):
        result = await auth_service.reset_password("invalid-token", "NewPass123")
        assert isinstance(result, Failure)
        error = result.failure()
        assert "Invalid or expired" in error.message


class TestAuthOAuth:
    """OAuth tests."""

    async def test_oauth_callback_new_user(self, auth_service: AuthService):
        with (
            patch("app.features.auth.service.fetch_oauth_userinfo") as mock_fetch,
            patch("app.features.auth.service.get_oauth_config") as mock_config,
            patch("app.features.auth.service.verify_oauth_state", return_value=True),
        ):
            mock_fetch.return_value = MagicMock(
                email="oauth@example.com",
                provider_user_id="oauth-123",
                full_name="OAuth User",
            )
            mock_config.return_value = MagicMock(
                client_id="client-id",
                authorization_endpoint="https://example.com/oauth/authorize",
                redirect_uri="https://example.com/callback",
                scope=["email", "profile"],
            )

            result = await auth_service.oauth_callback(
                provider="google",
                code="test-code",
                state="test-state",
                signed_state="signed-state",
            )
            tokens = result.unwrap()

            assert tokens.access_token is not None
            assert tokens.refresh_token is not None


# ─── In-Memory Store Behavior Tests ─────────────────────────────────────────────


class TestInMemoryAuthAccountStore:
    """Direct tests for InMemoryAuthAccountStore behavior."""

    async def test_find_by_email_returns_none_for_unknown(
        self, account_store: InMemoryAuthAccountStore
    ):
        result = await account_store.find_by_email("unknown@example.com")
        assert result.unwrap() is None

    async def test_find_by_email_returns_account(self, account_store: InMemoryAuthAccountStore):
        await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        result = await account_store.find_by_email("test@example.com")
        account = result.unwrap()
        assert account is not None
        assert account.email == "test@example.com"

    async def test_find_by_id(self, account_store: InMemoryAuthAccountStore):
        create_result = await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        created = create_result.unwrap()
        result = await account_store.find_by_id(created.id)
        account = result.unwrap()
        assert account is not None
        assert account.id == created.id
        assert "profile:read" in account.permissions
        assert "profile:write" in account.permissions

    async def test_create_duplicate_email_returns_conflict(
        self, account_store: InMemoryAuthAccountStore
    ):
        await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        result = await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass2"))
        )
        assert isinstance(result, Failure)

    async def test_replace_password_hash(self, account_store: InMemoryAuthAccountStore):
        create_result = await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        created = create_result.unwrap()
        old_hash = created.password_hash

        await account_store.replace_password_hash(created.id, "new_hashed_password")

        result = await account_store.find_by_id(created.id)
        updated = result.unwrap()
        assert updated.password_hash == "new_hashed_password"
        assert updated.password_hash != old_hash

    async def test_mark_email_verified(self, account_store: InMemoryAuthAccountStore):
        create_result = await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        created = create_result.unwrap()
        assert created.is_verified is False

        await account_store.mark_email_verified(created.id)

        result = await account_store.find_by_id(created.id)
        updated = result.unwrap()
        assert updated.is_verified is True
        assert updated.verification_token_hash is None

    async def test_set_and_consume_reset_token(self, account_store: InMemoryAuthAccountStore):
        create_result = await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        created = create_result.unwrap()

        token_hash = "test_reset_token_hash"
        expires_at = datetime.now(UTC) + timedelta(hours=1)

        await account_store.set_reset_token(created.id, token_hash, expires_at)

        result = await account_store.find_by_id(created.id)
        updated = result.unwrap()
        assert updated.reset_token_hash == token_hash
        assert updated.reset_token_expires_at is not None

        # Consume the token
        await account_store.consume_reset_token_and_replace_password(
            created.id, token_hash, "new_password_hash"
        )

        result = await account_store.find_by_id(created.id)
        consumed = result.unwrap()
        assert consumed.password_hash == "new_password_hash"
        assert consumed.reset_token_hash is None
        assert consumed.reset_token_expires_at is None

    async def test_consume_reset_token_with_wrong_hash_fails(
        self, account_store: InMemoryAuthAccountStore
    ):
        create_result = await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        created = create_result.unwrap()

        await account_store.set_reset_token(
            created.id, "correct_hash", datetime.now(UTC) + timedelta(hours=1)
        )

        result = await account_store.consume_reset_token_and_replace_password(
            created.id, "wrong_hash", "new_hash"
        )
        from returns.result import Failure

        assert isinstance(result, Failure)

    async def test_consume_expired_reset_token_fails(self, account_store: InMemoryAuthAccountStore):
        create_result = await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        created = create_result.unwrap()

        expired = datetime.now(UTC) - timedelta(hours=1)
        await account_store.set_reset_token(created.id, "token_hash", expired)

        result = await account_store.consume_reset_token_and_replace_password(
            created.id, "token_hash", "new_hash"
        )
        from returns.result import Failure

        assert isinstance(result, Failure)

    async def test_consume_reset_token_at_expiry_fails(
        self, account_store: InMemoryAuthAccountStore
    ):
        create_result = await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        created = create_result.unwrap()

        await account_store.set_reset_token(created.id, "token_hash", datetime.now(UTC))

        result = await account_store.consume_reset_token_and_replace_password(
            created.id, "token_hash", "new_hash"
        )
        assert isinstance(result, Failure)

    async def test_email_exists(self, account_store: InMemoryAuthAccountStore):
        assert (await account_store.email_exists("test@example.com")).unwrap() is False
        await account_store.create(
            NewAuthAccount(email="test@example.com", password_hash=hash_password("ValidPass1"))
        )
        assert (await account_store.email_exists("test@example.com")).unwrap() is True


async def test_account_store_profile_and_oauth_mutations_are_replayable() -> None:
    store = InMemoryAuthAccountStore()
    created = (
        await store.create(
            NewAuthAccount(
                email="test@example.com",
                password_hash=hash_password("ValidPass1"),
                oauth_account=OAuthAccountInput(
                    provider="google",
                    provider_user_id="google-1",
                    provider_email="test@example.com",
                ),
            )
        )
    ).unwrap()

    updated = (await store.update_profile(created.id, full_name="Ada Lovelace")).unwrap()
    assert updated.full_name == "Ada Lovelace"
    assert updated.oauth_accounts[0].provider_user_id == "google-1"

    await store.set_avatar_url(created.id, "https://cdn.example/avatar.png")
    await store.link_oauth_account(
        created.id,
        "github",
        "github-1",
        "test@example.com",
    )
    current = (await store.find_by_id(created.id)).unwrap()
    assert current is not None
    assert current.avatar_url == "https://cdn.example/avatar.png"
    assert current.oauth_providers == ("google", "github")

    # Linking the same provider identity is idempotent and does not duplicate it.
    await store.link_oauth_account(created.id, "github", "github-1", "test@example.com")
    replayed = (await store.find_by_id(created.id)).unwrap()
    assert replayed is not None
    assert replayed.oauth_providers == ("google", "github")


async def test_account_store_rejects_second_identity_for_same_oauth_provider() -> None:
    store = InMemoryAuthAccountStore()
    created = (
        await store.create(
            NewAuthAccount(
                email="test@example.com",
                password_hash=hash_password("ValidPass1"),
                oauth_account=OAuthAccountInput(
                    provider="google",
                    provider_user_id="google-1",
                ),
            )
        )
    ).unwrap()

    result = await store.link_oauth_account(created.id, "google", "google-2", None)

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthConflictError)
