"""Service-boundary regressions for refresh reuse (F4) and session listing (F9).

Reuse: a superseded refresh token must never mint tokens — the whole family
is revoked and the typed ``REFRESH_TOKEN_REUSE`` failure is returned, even
for the rotation's own successor token.

Listing: the production Redis client is not configured with
``decode_responses``, so session-index members arrive as bytes; the read
path must survive that (and must not destroy the index while misreading it).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi import Response
from returns.result import Failure, Success

from app.features.auth.adapters.persistence.mongodb.account_repository import (
    RefreshTokenRepository,
    SessionData,
)
from app.features.auth.adapters.persistence.mongodb.user_store import (
    InMemoryAuthAccountStore,
)
from app.features.auth.api.http.dto import LoginRequest, RegisterRequest
from app.features.auth.application.personal_tenant_provisioning import (
    PersonalTenantProvisioner,
)
from app.features.auth.application.service import AuthService
from app.features.auth.domain.errors import (
    AuthAuthenticationError,
    AuthCode,
    AuthRefreshReuseError,
)
from app.kernel.result import render_result
from app.security.tokens import decode_token
from app.shared.jobs.registry import UnregisteredTaskError

from .provisioning_fakes import FakeTenantSQLStore

if TYPE_CHECKING:
    from app.features.auth.api.http.dto import TokenResponse

pytestmark = pytest.mark.unit


def build_service(
    *, redis: FakeRedis | None = None
) -> tuple[AuthService, InMemoryAuthAccountStore, FakeRedis]:
    sql = FakeTenantSQLStore()
    accounts = InMemoryAuthAccountStore()
    redis = redis or FakeRedis(decode_responses=True)
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)
    service = AuthService(accounts, RefreshTokenRepository(redis), provisioner=provisioner)
    return service, accounts, redis


async def _login_verified_user(
    service: AuthService, accounts: InMemoryAuthAccountStore
) -> TokenResponse:
    registered = await service.register(
        RegisterRequest(email="reuse@example.com", password="ValidPass1")
    )
    assert isinstance(registered, Success)
    account = (await accounts.find_by_email("reuse@example.com")).unwrap()
    assert account is not None
    assert isinstance(await accounts.mark_email_verified(account.id), Success)
    login = await service.login(LoginRequest(email="reuse@example.com", password="ValidPass1"))
    assert isinstance(login, Success)
    return login.unwrap()


@pytest.mark.asyncio
async def test_first_refresh_still_rotates_and_succeeds() -> None:
    """The non-reuse path is unchanged: a fresh token rotates exactly once."""
    service, accounts, _redis = build_service()
    tokens = await _login_verified_user(service, accounts)

    refreshed = await service.refresh(tokens.refresh_token)
    assert isinstance(refreshed, Success)
    new_claims = decode_token(refreshed.unwrap().refresh_token)
    old_claims = decode_token(tokens.refresh_token)
    assert new_claims.refresh_generation == old_claims.refresh_generation + 1
    assert new_claims.family_id == old_claims.family_id


@pytest.mark.asyncio
async def test_reused_refresh_token_returns_typed_reuse_and_revokes_family() -> None:
    """Re-presenting a rotated token: typed 401 reuse and the family is dead."""
    service, accounts, _redis = build_service()
    tokens = await _login_verified_user(service, accounts)
    first = await service.refresh(tokens.refresh_token)
    assert isinstance(first, Success)

    reuse = await service.refresh(tokens.refresh_token)
    assert isinstance(reuse, Failure)
    assert isinstance(reuse.failure(), AuthRefreshReuseError)
    assert reuse.failure().code == AuthCode.REFRESH_TOKEN_REUSE

    # The whole family is revoked: even the NEW token is now rejected.
    successor = await service.refresh(first.unwrap().refresh_token)
    assert isinstance(successor, Failure)
    assert isinstance(successor.failure(), AuthAuthenticationError)
    assert successor.failure().message == "Session expired or revoked"


@pytest.mark.asyncio
async def test_login_lists_live_session_and_logout_removes_it() -> None:
    service, accounts, _redis = build_service()
    tokens = await _login_verified_user(service, accounts)
    claims = decode_token(tokens.access_token)
    assert claims.sid is not None

    listed = await service.list_sessions(user_id=claims.sub, current_session_id=claims.sid)
    assert isinstance(listed, Success)
    sessions = listed.unwrap()
    assert len(sessions) == 1
    session = sessions[0]
    assert session.session_id == claims.sid
    assert session.is_current is True
    assert session.device_id
    assert session.created_at <= datetime.now(UTC) + timedelta(minutes=1)
    assert session.expires_at > datetime.now(UTC)

    logout = await service.logout(tokens.refresh_token)
    assert isinstance(logout, Success)
    after = await service.list_sessions(user_id=claims.sub, current_session_id=claims.sid)
    assert isinstance(after, Success)
    assert after.unwrap() == []


@pytest.mark.asyncio
async def test_session_listing_survives_bytes_returning_redis_client() -> None:
    """The production client returns bytes members; listing must read and keep them."""
    redis = FakeRedis(decode_responses=False)
    repo = RefreshTokenRepository(redis)
    now = datetime.now(UTC)
    session = SessionData(
        session_id="session-bytes-1",
        user_id="507f1f77bcf86cd799439011",
        device_id="device-bytes-1",
        created_at=now,
        expires_at=now + timedelta(hours=1),
        ttl=3600,
        authz_revision=0,
        family_id="family-bytes-1",
    )
    assert isinstance(await repo.store_session(session), Success)

    listed = await repo.get_user_sessions("507f1f77bcf86cd799439011")
    assert isinstance(listed, Success)
    assert [s.session_id for s in listed.unwrap()] == ["session-bytes-1"]
    # The index is not destroyed by the read.
    remaining = await redis.smembers("auth:user_sessions:507f1f77bcf86cd799439011")
    assert remaining == {b"session-bytes-1"}


class _ExplodingSessionFactory:
    """Session-factory stand-in whose outbox seam raises the way production did.

    G1 regression: the refresh-reuse security-event emit targeted an event name
    absent from the job catalog, so the outbox publish raised
    ``UnregisteredTaskError`` (a Celery error, not a SQLAlchemy error) after the
    family had already been revoked — escaping the DB-error-only catch and
    replacing the typed reuse failure with a 500.
    """

    def __call__(self) -> None:
        message = "auth.refresh_token_reuse_detected"
        raise UnregisteredTaskError(
            message,
            known_names=frozenset({"auth.send_verification_email"}),
        )


@pytest.mark.asyncio
async def test_reuse_failure_survives_exploding_security_event_emit() -> None:
    """Reuse stays a typed 401 failure even when the audit emit raises (G1)."""
    sql = FakeTenantSQLStore()
    accounts = InMemoryAuthAccountStore()
    redis = FakeRedis(decode_responses=True)
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)
    service = AuthService(
        accounts,
        RefreshTokenRepository(redis),
        session_factory=_ExplodingSessionFactory(),  # type: ignore[arg-type]
        provisioner=provisioner,
    )
    tokens = await _login_verified_user(service, accounts)
    first = await service.refresh(tokens.refresh_token)
    assert isinstance(first, Success)

    reuse = await service.refresh(tokens.refresh_token)
    assert isinstance(reuse, Failure)
    assert isinstance(reuse.failure(), AuthRefreshReuseError)
    assert reuse.failure().code == AuthCode.REFRESH_TOKEN_REUSE

    # The failure renders as the typed 401 wire contract, never a 500.
    response = Response()
    rendered = render_result(reuse, response, message="Token refreshed")
    assert response.status_code == 401
    assert rendered.error is not None
    assert rendered.error.code == "REFRESH_TOKEN_REUSE"


@pytest.mark.asyncio
async def test_reuse_failure_survives_outbox_infrastructure_failure() -> None:
    """A DB failure on the audit emit also leaves the typed reuse failure intact."""
    from sqlalchemy.exc import OperationalError

    class _FailingSessionFactory:
        def __call__(self) -> None:
            raise OperationalError(
                statement="SELECT 1", params=None, orig=RuntimeError("connection lost")
            )

    sql = FakeTenantSQLStore()
    accounts = InMemoryAuthAccountStore()
    redis = FakeRedis(decode_responses=True)
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)
    service = AuthService(
        accounts,
        RefreshTokenRepository(redis),
        session_factory=_FailingSessionFactory(),  # type: ignore[arg-type]
        provisioner=provisioner,
    )
    tokens = await _login_verified_user(service, accounts)
    first = await service.refresh(tokens.refresh_token)
    assert isinstance(first, Success)

    reuse = await service.refresh(tokens.refresh_token)
    assert isinstance(reuse, Failure)
    assert isinstance(reuse.failure(), AuthRefreshReuseError)
    assert reuse.failure().code == AuthCode.REFRESH_TOKEN_REUSE
