"""Session-issuance invariant tests: no session without SQL membership.

Covers the approved provisioning flow end to end at the service boundary:
password registration, OAuth first-login creation, the typed pending semantics,
durable repair convergence, and refresh rejection for unprovisioned accounts.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fakeredis.aioredis import FakeRedis
from returns.result import Failure, Success

from app.features.auth.adapters.persistence.mongodb.account_repository import (
    RefreshTokenRepository,
)
from app.features.auth.adapters.persistence.mongodb.user_store import (
    InMemoryAuthAccountStore,
    NewAuthAccount,
)
from app.features.auth.api.http.dto import LoginRequest, RegisterRequest
from app.features.auth.application.oauth import OAuthUserInfo
from app.features.auth.application.personal_tenant_provisioning import (
    PersonalTenantProvisioner,
)
from app.features.auth.application.service import AuthService
from app.features.auth.domain.errors import (
    AuthInfrastructureError,
    AuthProvisioningPendingError,
)
from app.features.auth.domain.tenancy import personal_tenant_id
from app.security.passwords import hash_password

from .provisioning_fakes import FakeTenantSQLStore


def build_service() -> tuple[
    AuthService, FakeTenantSQLStore, InMemoryAuthAccountStore, FakeRedis
]:
    sql = FakeTenantSQLStore()
    accounts = InMemoryAuthAccountStore()
    redis = FakeRedis(decode_responses=True)
    provisioner = PersonalTenantProvisioner(sql.factory, accounts)
    service = AuthService(accounts, RefreshTokenRepository(redis), provisioner=provisioner)
    return service, sql, accounts, redis


async def verify_account(accounts: InMemoryAuthAccountStore, email: str) -> str:
    account = (await accounts.find_by_email(email)).unwrap()
    assert account is not None
    assert isinstance(await accounts.mark_email_verified(account.id), Success)
    return account.id


@pytest.mark.asyncio
async def test_register_verify_login_issues_session_without_any_migration() -> None:
    service, sql, accounts, _redis = build_service()

    registered = await service.register(
        RegisterRequest(email="fresh@example.com", password="ValidPass1")
    )
    assert isinstance(registered, Success)
    user = registered.unwrap()
    # The SQL personal tenant and membership already exist — no migration ran.
    assert personal_tenant_id(user.id) in sql.tenants
    assert (personal_tenant_id(user.id), user.id) in sql.memberships

    await verify_account(accounts, "fresh@example.com")
    login = await service.login(LoginRequest(email="fresh@example.com", password="ValidPass1"))
    assert isinstance(login, Success)
    tokens = login.unwrap()
    assert tokens.access_token
    assert tokens.refresh_token

    refreshed = await service.refresh(tokens.refresh_token)
    assert isinstance(refreshed, Success)


@pytest.mark.asyncio
async def test_oauth_first_login_creates_account_session_and_sql_membership() -> None:
    service, sql, accounts, _redis = build_service()

    with (
        patch("app.features.auth.application.service.verify_oauth_state", return_value=True),
        patch("app.features.auth.application.service.get_oauth_config"),
        patch(
            "app.features.auth.application.service.fetch_oauth_userinfo",
            return_value=OAuthUserInfo(
                email="oauth-new@example.com",
                provider_user_id="google-1",
                full_name="OAuth User",
                email_verified=True,
            ),
        ),
    ):
        result = await service.oauth_callback("google", "code", "state", "signed")

    assert isinstance(result, Success)
    tokens = result.unwrap()
    assert tokens.access_token
    account = (await accounts.find_by_email("oauth-new@example.com")).unwrap()
    assert account is not None
    assert (personal_tenant_id(account.id), account.id) in sql.memberships


@pytest.mark.asyncio
async def test_sql_provisioning_failure_keeps_account_pending_and_issues_no_session() -> None:
    service, sql, accounts, redis = build_service()
    sql.fail_next_commit = True

    registered = await service.register(
        RegisterRequest(email="pending@example.com", password="ValidPass1")
    )

    assert isinstance(registered, Failure)
    assert isinstance(registered.failure(), AuthInfrastructureError)
    account = (await accounts.find_by_email("pending@example.com")).unwrap()
    assert account is not None
    assert account.provisioning_state.value == "pending"
    assert not sql.tenants
    assert not sql.memberships

    # Correct credentials while Postgres is down → retryable infrastructure
    # failure, never invalid credentials, and never a session.
    await verify_account(accounts, "pending@example.com")
    sql.fail_next_commit = True  # Postgres is still unreachable for the repair.
    login = await service.login(
        LoginRequest(email="pending@example.com", password="ValidPass1")
    )
    assert isinstance(login, Failure)
    assert isinstance(login.failure(), AuthInfrastructureError)
    assert await redis.keys("auth:session:*") == []


@pytest.mark.asyncio
async def test_durable_pending_account_repairs_on_next_login() -> None:
    service, sql, accounts, _redis = build_service()
    sql.fail_next_commit = True
    registered = await service.register(
        RegisterRequest(email="repair@example.com", password="ValidPass1")
    )
    assert isinstance(registered, Failure)
    user_id = await verify_account(accounts, "repair@example.com")

    # Postgres recovered: the idempotent durable repair converges on login.
    login = await service.login(
        LoginRequest(email="repair@example.com", password="ValidPass1")
    )

    assert isinstance(login, Success)
    assert (personal_tenant_id(user_id), user_id) in sql.memberships
    membership = sql.memberships[personal_tenant_id(user_id), user_id]
    assert membership.mirror_sync_pending is False


@pytest.mark.asyncio
async def test_legacy_unprovisioned_account_gets_typed_pending_not_invalid_credentials() -> None:
    service, sql, accounts, _redis = build_service()
    # Legacy account: created before provisioning existed, never migrated.
    created = await accounts.create(
        NewAuthAccount(
            email="legacy@example.com",
            password_hash=hash_password("ValidPass1"),
            is_verified=True,
        )
    )
    assert isinstance(created, Success)

    login = await service.login(LoginRequest(email="legacy@example.com", password="ValidPass1"))

    assert isinstance(login, Failure)
    error = login.failure()
    assert isinstance(error, AuthProvisioningPendingError)
    assert error.code.value == "ACCOUNT_PROVISIONING_PENDING"
    assert error.retryable is True
    # No SQL state was lazily created by the login path.
    assert not sql.tenants
    assert not sql.memberships


@pytest.mark.asyncio
async def test_refresh_rejects_account_whose_sql_membership_disappeared() -> None:
    service, sql, accounts, _redis = build_service()
    registered = await service.register(
        RegisterRequest(email="loss@example.com", password="ValidPass1")
    )
    assert isinstance(registered, Success)
    await verify_account(accounts, "loss@example.com")
    login = await service.login(LoginRequest(email="loss@example.com", password="ValidPass1"))
    assert isinstance(login, Success)
    tokens = login.unwrap()

    # Simulate the SQL membership being lost: refresh must fail with the typed
    # pending semantics — never invalid-credentials, never a new session.
    sql.memberships.clear()
    sql.tenants.clear()

    refreshed = await service.refresh(tokens.refresh_token)

    assert isinstance(refreshed, Failure)
    error = refreshed.failure()
    assert isinstance(error, AuthProvisioningPendingError)
    assert error.retryable is True


@pytest.mark.asyncio
async def test_refresh_with_sql_unavailable_returns_retryable_infrastructure_error() -> None:
    service, sql, accounts, _redis = build_service()
    registered = await service.register(
        RegisterRequest(email="outage@example.com", password="ValidPass1")
    )
    assert isinstance(registered, Success)
    await verify_account(accounts, "outage@example.com")
    login = await service.login(LoginRequest(email="outage@example.com", password="ValidPass1"))
    assert isinstance(login, Success)
    tokens = login.unwrap()

    sql.fail_select = True
    refreshed = await service.refresh(tokens.refresh_token)

    assert isinstance(refreshed, Failure)
    assert isinstance(refreshed.failure(), AuthInfrastructureError)
    assert refreshed.failure().retryable is True


@pytest.mark.asyncio
async def test_register_retry_completes_provisioning_for_pending_account() -> None:
    """A pending account's second registration retries provisioning instead of dead-ending."""
    service, sql, accounts, _redis = build_service()
    sql.fail_next_commit = True

    first = await service.register(RegisterRequest(email="retry@example.com", password="ValidPass1"))
    assert isinstance(first, Failure)
    assert isinstance(first.failure(), AuthInfrastructureError)
    sql.fail_next_commit = False

    second = await service.register(
        RegisterRequest(email="retry@example.com", password="ValidPass1")
    )
    assert isinstance(second, Success)
    account = (await accounts.find_by_email("retry@example.com")).unwrap()
    assert account is not None
    assert (personal_tenant_id(account.id), account.id) in sql.memberships

    # Once the lifecycle is no longer pending, the conflict semantics return.
    third = await service.register(
        RegisterRequest(email="retry@example.com", password="ValidPass1")
    )
    assert isinstance(third, Failure)
    assert third.failure().code.value == "USER_CONFLICT"
