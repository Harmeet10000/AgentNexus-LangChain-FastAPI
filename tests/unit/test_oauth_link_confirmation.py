"""Security contracts for explicit OAuth account linking."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from fakeredis.aioredis import FakeRedis
from returns.result import Failure, Success

from app.features.auth.account_store import InMemoryAuthAccountStore, NewAuthAccount
from app.features.auth.exceptions import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
)
from app.features.auth.oauth import OAuthUserInfo
from app.features.auth.repository import RefreshTokenRepository, SessionData
from app.features.auth.service import AuthService
from app.security.passwords import hash_password
from app.security.tokens import TokenClaims

pytestmark = pytest.mark.unit


def _claims(user_id: str, session_id: str, revision: int = 0) -> TokenClaims:
    return TokenClaims(
        sub=user_id,
        jti="jti",
        sid=session_id,
        role="user",
        permissions=[],
        token_type="access",
        authz_revision=revision,
    )


async def _account_and_session() -> tuple[
    InMemoryAuthAccountStore, RefreshTokenRepository, str, str
]:
    store = InMemoryAuthAccountStore()
    account = (
        await store.create(
            NewAuthAccount(
                email="local@example.com",
                password_hash=hash_password("ValidPass1"),
                is_verified=True,
            )
        )
    ).unwrap()
    redis = FakeRedis(decode_responses=True)
    repo = RefreshTokenRepository(redis)
    session_id = "session-1"
    now = datetime.now(UTC)
    assert isinstance(
        await repo.store_session(
            SessionData(
                session_id=session_id,
                user_id=account.id,
                device_id="device-1",
                created_at=now,
                expires_at=now + timedelta(hours=1),
                ttl=3600,
                authz_revision=account.role_revision,
            )
        ),
        Success,
    )
    return store, repo, account.id, session_id


async def test_public_oauth_callback_never_links_by_email() -> None:
    store, repo, account_id, _ = await _account_and_session()
    service = AuthService(store, repo)
    with (
        patch("app.features.auth.service.verify_oauth_state", return_value=True),
        patch("app.features.auth.service.get_oauth_config"),
        patch(
            "app.features.auth.service.fetch_oauth_userinfo",
            return_value=OAuthUserInfo(
                email="local@example.com",
                provider_user_id="google-subject",
                email_verified=True,
            ),
        ),
    ):
        result = await service.oauth_callback("google", "code", "state", "signed")
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthConflictError)
    account = (await store.find_by_id(account_id)).unwrap()
    assert account is not None
    assert account.oauth_accounts == ()


async def test_link_callback_requires_confirmation_and_consumes_action_once() -> None:
    store, repo, account_id, session_id = await _account_and_session()
    service = AuthService(store, repo)
    state = "oauth-state"
    assert isinstance(
        await repo.create_oauth_link_intent(
            user_id=account_id,
            session_id=session_id,
            provider="google",
            authz_revision=0,
            state=state,
        ),
        Success,
    )
    with (
        patch("app.features.auth.service.verify_oauth_state", return_value=True),
        patch("app.features.auth.service.get_oauth_config"),
        patch(
            "app.features.auth.service.fetch_oauth_userinfo",
            return_value=OAuthUserInfo(
                email="provider@example.com",
                provider_user_id="google-subject",
                full_name="Provider User",
                email_verified=True,
            ),
        ),
    ):
        callback_result = await service.oauth_callback("google", "code", state, "signed")
    assert isinstance(callback_result, Success)
    candidate = callback_result.unwrap()
    assert candidate.provider_email == "provider@example.com"
    assert (await store.find_by_id(account_id)).unwrap().oauth_accounts == ()
    preview = await service.oauth_link_preview(
        candidate.action_token, user_id=account_id, session_id=session_id
    )
    assert isinstance(preview, Success)
    confirm = await service.oauth_link_confirm(
        candidate.action_token,
        user_id=account_id,
        session_id=session_id,
        authz_revision=0,
        confirm=True,
    )
    assert isinstance(confirm, Success)
    account = (await store.find_by_id(account_id)).unwrap()
    assert account is not None
    assert account.oauth_accounts[0].provider_user_id == "google-subject"
    replay = await service.oauth_link_confirm(
        candidate.action_token,
        user_id=account_id,
        session_id=session_id,
        authz_revision=0,
        confirm=True,
    )
    assert isinstance(replay, Failure)
    assert isinstance(replay.failure(), AuthAuthenticationError)


async def test_link_preview_rejects_wrong_session_and_expired_state() -> None:
    store, repo, account_id, session_id = await _account_and_session()
    service = AuthService(store, repo)
    state = "state-for-expiry-test"
    await repo.create_oauth_link_intent(
        user_id=account_id,
        session_id=session_id,
        provider="github",
        authz_revision=0,
        state=state,
        ttl=1,
    )
    completed = await repo.complete_oauth_link_intent(
        state=state,
        provider_user_id="github-subject",
        provider_email="provider@example.com",
        full_name=None,
    )
    assert isinstance(completed, Success)
    _, action_token = completed.unwrap()
    wrong_session = await service.oauth_link_preview(
        action_token, user_id=account_id, session_id="other-session"
    )
    assert isinstance(wrong_session, Failure)
    assert isinstance(wrong_session.failure(), AuthAuthorizationError)
    await repo._redis.expire(f"auth:oauth_link_action:{repo._digest(action_token)}", 0)
    expired = await service.oauth_link_preview(
        action_token, user_id=account_id, session_id=session_id
    )
    assert isinstance(expired, Failure)
    assert isinstance(expired.failure(), AuthAuthenticationError)


async def test_link_callback_rejects_provider_mixup_and_old_session() -> None:
    store, repo, account_id, session_id = await _account_and_session()
    service = AuthService(store, repo)
    await repo.create_oauth_link_intent(
        user_id=account_id,
        session_id=session_id,
        provider="google",
        authz_revision=0,
        state="provider-bound-state",
    )
    with (
        patch("app.features.auth.service.verify_oauth_state", return_value=True),
        patch("app.features.auth.service.get_oauth_config"),
        patch(
            "app.features.auth.service.fetch_oauth_userinfo",
            return_value=OAuthUserInfo(
                email="provider@example.com",
                provider_user_id="github-subject",
                email_verified=True,
            ),
        ),
    ):
        result = await service.oauth_callback("github", "code", "provider-bound-state", "signed")
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthAuthenticationError)

    old_session = SessionData(
        session_id="old-session",
        user_id=account_id,
        device_id="device-old",
        created_at=datetime.now(UTC) - timedelta(hours=1),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        ttl=3600,
        authz_revision=0,
    )
    assert isinstance(await repo.store_session(old_session), Success)
    old_start = await service.oauth_link_get_authorization_url(
        "google", user_id=account_id, session_id="old-session", authz_revision=0
    )
    assert isinstance(old_start, Failure)
    assert isinstance(old_start.failure(), AuthAuthenticationError)


async def test_concurrent_confirmation_has_one_winner() -> None:
    store, repo, account_id, session_id = await _account_and_session()
    service = AuthService(store, repo)
    await repo.create_oauth_link_intent(
        user_id=account_id,
        session_id=session_id,
        provider="google",
        authz_revision=0,
        state="concurrent-state",
    )
    completed = await repo.complete_oauth_link_intent(
        state="concurrent-state",
        provider_user_id="google-concurrent",
        provider_email="provider@example.com",
        full_name=None,
    )
    assert isinstance(completed, Success)
    _, action_token = completed.unwrap()
    results = await asyncio.gather(
        *(
            service.oauth_link_confirm(
                action_token,
                user_id=account_id,
                session_id=session_id,
                authz_revision=0,
                confirm=True,
            )
            for _ in range(2)
        )
    )
    assert sum(isinstance(result, Success) for result in results) == 1
    assert sum(isinstance(result, Failure) for result in results) == 1


async def test_unlink_refuses_to_remove_last_oauth_sign_in_method() -> None:
    store = InMemoryAuthAccountStore()
    created = (
        await store.create(
            NewAuthAccount(
                email="oauth-only@example.com",
                password_hash=None,
                is_verified=True,
                oauth_account={
                    "provider": "google",
                    "provider_user_id": "google-only",
                    "provider_email": "oauth-only@example.com",
                },
            )
        )
    ).unwrap()
    result = await store.unlink_oauth_account(created.id, "google")
    assert isinstance(result, Failure)
    assert isinstance(result.failure(), AuthConflictError)
