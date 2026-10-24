"""Regression tests for Redis session authority and OAuth trust boundaries."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis.aioredis import FakeRedis
from returns.result import Failure, Success

from app.features.auth.account_store import InMemoryAuthAccountStore, NewAuthAccount
from app.features.auth.dependencies import get_token_claims, require_permission, require_role
from app.features.auth.dto import SessionResponse
from app.features.auth.exceptions import AuthAuthenticationError, AuthInfrastructureError
from app.features.auth.models import Permission, UserRole
from app.features.auth.oauth import OAuthProviderConfig, OAuthUserInfo, fetch_oauth_userinfo
from app.features.auth.repository import RefreshTokenRepository, SessionData
from app.features.auth.service import AuthService
from app.features.auth.token_audit_log import TokenAuditLog
from app.features.users.service import UserAdminService
from app.security.tokens import TokenClaims
from app.utils import ForbiddenException, InfrastructureException, UnauthorizedException

pytestmark = pytest.mark.unit


def _claims(*, sid: str | None = "session-1", sub: str = "user-1") -> TokenClaims:
    return TokenClaims(
        sub=sub,
        jti="jti-1",
        sid=sid,
        role="user",
        permissions=[],
        token_type="access",
    )


def _session(session_id: str, user_id: str = "user-1") -> SessionData:
    now = datetime.now(UTC)
    return SessionData(
        session_id=session_id,
        user_id=user_id,
        device_id=f"device-{session_id}",
        created_at=now,
        expires_at=now + timedelta(hours=1),
        ttl=3600,
        user_agent="test-agent",
    )


async def test_http_claims_fail_closed_when_redis_session_is_missing() -> None:
    repo = AsyncMock()
    repo.get_session.return_value = Success(None)
    with (
        patch("app.features.auth.dependencies.decode_token", return_value=_claims()),
        pytest.raises(UnauthorizedException, match="Session expired or revoked"),
    ):
        await get_token_claims("raw-token", repo)


async def test_http_claims_skip_redis_for_stateless_impersonation() -> None:
    repo = AsyncMock()
    with patch("app.features.auth.dependencies.decode_token", return_value=_claims(sid=None)):
        result = await get_token_claims("raw-token", repo)
    assert result.sid is None
    repo.get_session.assert_not_awaited()


async def test_http_claims_fail_closed_when_redis_verification_fails() -> None:
    repo = AsyncMock()
    repo.get_session.return_value = Failure(RuntimeError("redis unavailable"))
    with (
        patch("app.features.auth.dependencies.decode_token", return_value=_claims()),
        pytest.raises(UnauthorizedException, match="Unable to verify session"),
    ):
        await get_token_claims("raw-token", repo)


async def test_http_claims_surfaces_redis_outage_as_retryable_failure() -> None:
    repo = AsyncMock()
    repo.get_session.return_value = Failure(
        AuthInfrastructureError(message="redis unavailable", source="test")
    )
    with (
        patch("app.features.auth.dependencies.decode_token", return_value=_claims()),
        pytest.raises(InfrastructureException, match="Session store unavailable"),
    ):
        await get_token_claims("raw-token", repo)


async def test_malformed_redis_session_fails_closed() -> None:
    redis = FakeRedis(decode_responses=True)
    repo = RefreshTokenRepository(redis)
    await redis.set("auth:session:session-1", "not-json")

    result = await repo.get_session("session-1")

    assert isinstance(result, Failure)


@pytest.mark.parametrize(
    ("session", "message"),
    [
        (_session("session-1", user_id="other-user"), "Session expired or revoked"),
        (
            SessionData(
                session_id="session-1",
                user_id="user-1",
                device_id="device-1",
                created_at=datetime.now(UTC) - timedelta(hours=2),
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
                ttl=3600,
            ),
            "Session expired or revoked",
        ),
    ],
)
async def test_http_claims_validate_session_owner_and_expiry(
    session: SessionData,
    message: str,
) -> None:
    repo = AsyncMock()
    repo.get_session.return_value = Success(session)
    with (
        patch("app.features.auth.dependencies.decode_token", return_value=_claims()),
        pytest.raises(UnauthorizedException, match=message),
    ):
        await get_token_claims("raw-token", repo)


async def test_revoke_all_is_atomic_and_returns_exact_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = FakeRedis(decode_responses=True)
    repo = RefreshTokenRepository(redis)
    original_eval = redis.eval

    async def traced_eval(*args: object, **kwargs: object) -> object:
        return await original_eval(*args, **kwargs)

    eval_call = AsyncMock(side_effect=traced_eval)
    monkeypatch.setattr(redis, "eval", eval_call)
    await redis.sadd("auth:user_sessions:user-1", "session-1", "session-2", "session-3")
    for session_id in ("session-1", "session-2", "session-3"):
        await redis.set(
            f"auth:session:{session_id}",
            json.dumps(_session(session_id).model_dump(mode="json")),
        )

    class _AuditQuery:
        async def update(self, _update: object) -> None:
            return None

    monkeypatch.setattr(TokenAuditLog, "find", lambda *_args: _AuditQuery())
    result = await repo.revoke_all_user_sessions(
        "user-1", except_session_id="session-2", reason="test"
    )

    assert isinstance(result, Success)
    assert set(result.unwrap()) == {"session-1", "session-3"}
    assert await redis.get("auth:session:session-1") is None
    assert await redis.get("auth:session:session-3") is None
    assert await redis.get("auth:session:session-2") is not None
    assert await redis.smembers("auth:user_sessions:user-1") == {"session-2"}
    eval_call.assert_awaited_once()
    script = eval_call.await_args.args[0]
    assert "SMEMBERS" in script
    assert "DEL" in script
    assert "SREM" in script


async def test_mongo_audit_failure_does_not_report_failed_redis_revoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis(decode_responses=True)
    repo = RefreshTokenRepository(redis)
    await redis.sadd("auth:user_sessions:user-1", "session-1")
    await redis.set("auth:session:session-1", "active")
    audit_called = False

    class _AuditQuery:
        async def update(self, _update: object) -> None:
            nonlocal audit_called
            audit_called = True
            from pymongo.errors import PyMongoError

            msg = "mongo unavailable"
            raise PyMongoError(msg)

    monkeypatch.setattr(TokenAuditLog, "find", lambda *_args: _AuditQuery())
    result = await repo.revoke_all_user_sessions("user-1", reason="test")

    assert isinstance(result, Success)
    assert result.unwrap() == ["session-1"]
    assert await redis.get("auth:session:session-1") is None
    assert audit_called


async def test_mongo_store_audit_failure_does_not_report_failed_redis_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis(decode_responses=True)
    repo = RefreshTokenRepository(redis)

    async def fail_insert() -> None:
        from pymongo.errors import PyMongoError

        msg = "mongo unavailable"
        raise PyMongoError(msg)

    audit_insert = AsyncMock(side_effect=fail_insert)
    monkeypatch.setattr(TokenAuditLog.return_value, "insert", audit_insert)
    session = _session("session-1", user_id="507f1f77bcf86cd799439011")
    result = await repo.store_session(session)

    assert isinstance(result, Success)
    assert await redis.get("auth:session:session-1") is not None
    audit_insert.assert_awaited_once()


async def test_session_index_ttl_never_shrinks_for_shorter_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis(decode_responses=True)
    repo = RefreshTokenRepository(redis)
    monkeypatch.setattr(TokenAuditLog.return_value, "insert", AsyncMock(return_value=None))
    long_session = _session("long", user_id="507f1f77bcf86cd799439011")
    short_session = long_session.model_copy(update={"session_id": "short", "ttl": 30}, deep=True)

    assert isinstance((await repo.store_session(long_session)), Success)
    long_ttl = await redis.ttl("auth:user_sessions:507f1f77bcf86cd799439011")
    assert isinstance((await repo.store_session(short_session)), Success)
    short_ttl = await redis.ttl("auth:user_sessions:507f1f77bcf86cd799439011")

    assert long_ttl > 30
    assert short_ttl > 30


async def test_mongo_single_revoke_audit_failure_does_not_report_failed_redis_revoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = FakeRedis(decode_responses=True)
    repo = RefreshTokenRepository(redis)
    await redis.sadd("auth:user_sessions:user-1", "session-1")
    await redis.set("auth:session:session-1", "active")

    class _AuditQuery:
        async def update(self, _update: object) -> None:
            from pymongo.errors import PyMongoError

            msg = "mongo unavailable"
            raise PyMongoError(msg)

    monkeypatch.setattr(TokenAuditLog, "find_one", lambda *_args: _AuditQuery())
    result = await repo.revoke_session("session-1", "user-1", reason="test")

    assert isinstance(result, Success)
    assert await redis.get("auth:session:session-1") is None


async def test_verified_provider_cannot_link_unverified_local_account() -> None:
    account_store = InMemoryAuthAccountStore()
    created = await account_store.create(
        NewAuthAccount(email="user@example.com", password_hash="hash", is_verified=False)
    )
    account = created.unwrap()
    service = AuthService(account_store, AsyncMock())

    with (
        patch("app.features.auth.service.verify_oauth_state", return_value=True),
        patch("app.features.auth.service.get_oauth_config"),
        patch(
            "app.features.auth.service.fetch_oauth_userinfo",
            return_value=OAuthUserInfo(
                email=account.email,
                provider_user_id="google-1",
                email_verified=True,
            ),
        ),
    ):
        result = await service.oauth_callback("google", "code", "state", "signed")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthAuthenticationError)
    current = (await account_store.find_by_id(account.id)).unwrap()
    assert current is not None
    assert current.oauth_accounts == ()


async def test_unverified_provider_cannot_link_verified_local_account() -> None:
    account_store = InMemoryAuthAccountStore()
    created = await account_store.create(
        NewAuthAccount(email="user@example.com", password_hash="hash", is_verified=True)
    )
    account = created.unwrap()
    service = AuthService(account_store, AsyncMock())

    with (
        patch("app.features.auth.service.verify_oauth_state", return_value=True),
        patch("app.features.auth.service.get_oauth_config"),
        patch(
            "app.features.auth.service.fetch_oauth_userinfo",
            return_value=OAuthUserInfo(
                email=account.email,
                provider_user_id="google-1",
                email_verified=False,
            ),
        ),
    ):
        result = await service.oauth_callback("google", "code", "state", "signed")

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthAuthenticationError)
    current = (await account_store.find_by_id(account.id)).unwrap()
    assert current is not None
    assert current.oauth_accounts == ()


async def test_google_unverified_email_cannot_create_account() -> None:
    account_store = InMemoryAuthAccountStore()
    service = AuthService(account_store, AsyncMock())
    with (
        patch("app.features.auth.service.verify_oauth_state", return_value=True),
        patch("app.features.auth.service.get_oauth_config"),
        patch(
            "app.features.auth.service.fetch_oauth_userinfo",
            return_value=OAuthUserInfo(
                email="new@example.com",
                provider_user_id="google-1",
                email_verified=False,
            ),
        ),
    ):
        result = await service.oauth_callback("google", "code", "state", "signed")
    assert isinstance(result, Failure)
    assert (await account_store.find_by_email("new@example.com")).unwrap() is None


async def test_github_userinfo_uses_primary_verified_email_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        def __init__(self, payload: object) -> None:
            self._payload = payload

        def json(self) -> object:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self) -> None:
            self.urls: list[str] = []

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def fetch_token(self, *_args: object, **_kwargs: object) -> None:
            return None

        async def get(self, url: str, **_kwargs: object) -> _Response:
            self.urls.append(url)
            if url.endswith("/user"):
                return _Response({"id": 42, "email": "untrusted@example.com", "name": "User"})
            return _Response(
                [
                    {"email": "primary-unverified@example.com", "primary": True, "verified": False},
                    {"email": "other@example.com", "primary": False, "verified": True},
                    {"email": "verified@example.com", "primary": True, "verified": True},
                ]
            )

    client = _Client()
    monkeypatch.setattr("app.features.auth.oauth.AsyncOAuth2Client", lambda **_kwargs: client)
    config = OAuthProviderConfig(
        client_id="client",
        client_secret="secret",
        authorization_endpoint="https://example.com/authorize",
        token_endpoint="https://example.com/token",
        scope="read:user user:email",
        redirect_uri="https://example.com/callback",
    )

    result = await fetch_oauth_userinfo("github", config, "code")

    assert result.email == "verified@example.com"
    assert result.email_verified is True
    assert client.urls == ["https://api.github.com/user", "https://api.github.com/user/emails"]


def test_session_response_exposes_user_agent() -> None:
    response = SessionResponse(
        session_id="session-1",
        device_id="device-1",
        user_agent="Mozilla/5.0",
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    assert response.model_dump(by_alias=True)["userAgent"] == "Mozilla/5.0"


async def test_role_change_revokes_target_sessions() -> None:
    user = type(
        "User",
        (),
        {
            "id": "target",
            "email": "target@example.com",
            "full_name": None,
            "role": UserRole.USER,
            "is_verified": True,
            "is_active": True,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "oauth_accounts": [],
        },
    )()
    user_repo = type(
        "UserRepo",
        (),
        {
            "find_by_id": AsyncMock(return_value=Success(user)),
            "update_role": AsyncMock(return_value=Success(user)),
        },
    )()
    token_repo = type(
        "TokenRepo",
        (),
        {"revoke_all_user_sessions": AsyncMock(return_value=Success([]))},
    )()
    service = UserAdminService(user_repo, token_repo)

    result = await service.update_role("target", UserRole.ADMIN, "admin-user")

    assert isinstance(result, Success)
    token_repo.revoke_all_user_sessions.assert_awaited_once_with(
        user_id="target", reason="role_changed"
    )


async def test_noop_role_change_does_not_revoke_sessions() -> None:
    user = type(
        "User",
        (),
        {
            "id": "target",
            "email": "target@example.com",
            "full_name": None,
            "role": UserRole.USER,
            "is_verified": True,
            "is_active": True,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "oauth_accounts": [],
        },
    )()
    user_repo = type(
        "UserRepo",
        (),
        {
            "find_by_id": AsyncMock(return_value=Success(user)),
            "update_role": AsyncMock(return_value=Success(user)),
        },
    )()
    token_repo = type(
        "TokenRepo",
        (),
        {"revoke_all_user_sessions": AsyncMock(return_value=Success([]))},
    )()
    service = UserAdminService(user_repo, token_repo)

    result = await service.update_role("target", UserRole.USER, "admin-user")

    assert isinstance(result, Success)
    user_repo.update_role.assert_not_awaited()
    token_repo.revoke_all_user_sessions.assert_not_awaited()


async def test_rbac_guards_use_live_account_permissions_and_role() -> None:
    account = type(
        "Account",
        (),
        {
            "is_active": True,
            "permissions": frozenset(),
            "role": UserRole.USER,
        },
    )()

    with pytest.raises(ForbiddenException):
        await require_permission(Permission.USERS_WRITE)(_claims(), account)
    with pytest.raises(ForbiddenException):
        await require_role(UserRole.ADMIN)(_claims(), account)
