from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest
from returns.result import Failure, Success

import app.features.auth.public as auth_public
from app.features.auth.domain.errors import (
    AuthAuthenticationError,
    AuthInfrastructureError,
    AuthSessionNotFoundError,
)


@dataclass
class FakeAccount:
    id: str = "account-1"
    email: str = "account@example.com"
    full_name: str | None = "Account One"
    avatar_url: str | None = None
    locale: str | None = "en"
    is_verified: bool = True
    is_active: bool = True
    role: str = "user"
    created_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    updated_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    password_hash: str | None = "old-hash"
    role_revision: int = 0
    legal_roles: tuple[str, ...] = ("analyst",)
    permissions: frozenset[str] = frozenset({"profile:read"})
    session_revocation_pending: bool = False


@dataclass
class FakeSession:
    session_id: str
    user_id: str
    family_id: str | None = None


class FakeAccountStore:
    def __init__(self, account: FakeAccount | None = None) -> None:
        self.account = account or FakeAccount()
        self.failure_operation: str | None = None
        self.replaced_hash: str | None = None

    def _failure(self, operation: str) -> Failure[AuthInfrastructureError] | None:
        if self.failure_operation == operation:
            return Failure(
                AuthInfrastructureError(
                    message=f"{operation} failed",
                    details={"operation": operation},
                    source="fake",
                )
            )
        return None

    async def find_by_id(self, account_id: str):
        if failure := self._failure("read"):
            return failure
        return Success(self.account if account_id == self.account.id else None)

    async def find_by_email(self, email: str):
        if failure := self._failure("read"):
            return failure
        return Success(self.account if email.lower() == self.account.email else None)

    async def update_profile(self, account_id: str, *, full_name: str):
        if failure := self._failure("profile"):
            return failure
        self.account = replace(self.account, full_name=full_name)
        return Success(self.account)

    async def set_avatar_url(self, account_id: str, avatar_url: str | None):
        if failure := self._failure("profile"):
            return failure
        self.account = replace(self.account, avatar_url=avatar_url)
        return Success(None)

    async def set_locale(self, account_id: str, locale: str):
        if failure := self._failure("profile"):
            return failure
        self.account = replace(self.account, locale=locale)
        return Success(None)

    async def replace_password_hash(self, account_id: str, password_hash: str):
        if failure := self._failure("password"):
            return failure
        self.replaced_hash = password_hash
        self.account = replace(self.account, password_hash=password_hash)
        return Success(None)


class FakeSessionPort:
    def __init__(self) -> None:
        self.sessions = {"session-1": FakeSession("session-1", "account-1", "family-1")}
        self.failure = False
        self.revoked: list[str] = []

    async def get_session(self, session_id: str):
        if self.failure:
            return Failure(
                AuthInfrastructureError(message="session lookup failed", details={}, source="fake")
            )
        return Success(self.sessions.get(session_id))

    async def revoke_refresh_family(
        self, family_id: str, *, reason: str, user_id: str, session_id: str
    ):
        if self.failure:
            return Failure(
                AuthInfrastructureError(message="family revoke failed", details={}, source="fake")
            )
        return Success(None)

    async def revoke_session(self, session_id: str, user_id: str, reason: str = "logout"):
        if self.failure:
            return Failure(
                AuthInfrastructureError(message="session revoke failed", details={}, source="fake")
            )
        self.revoked.append(session_id)
        self.sessions.pop(session_id, None)
        return Success(None)

    async def revoke_all_user_sessions(
        self,
        user_id: str,
        except_session_id: str | None = None,
        reason: str = "revoke_all",
        authz_revision: int | None = None,
    ):
        if self.failure:
            return Failure(
                AuthInfrastructureError(
                    message="all-session revoke failed", details={}, source="fake"
                )
            )
        revoked = [
            session_id
            for session_id, session in self.sessions.items()
            if session.user_id == user_id and session_id != except_session_id
        ]
        self.revoked.extend(revoked)
        return Success(revoked)


class FakeHasher:
    def verify(self, hashed: str, plain: str) -> bool:
        return hashed == "old-hash" and plain == "CurrentPass1"

    def hash(self, password: str) -> str:
        return f"hash:{password}"


class FakeAdministration:
    def __init__(self, account: FakeAccount) -> None:
        self.account = account

    async def update_role(self, account_id: str, role: str):
        self.account = replace(
            self.account,
            role=role,
            role_revision=self.account.role_revision + 1,
            session_revocation_pending=True,
        )
        return Success(self.account)

    async def set_active(self, account_id: str, *, is_active: bool):
        self.account = replace(self.account, is_active=is_active)
        return Success(self.account)

    async def clear_session_revocation_pending(self, account_id: str, *, expected_revision: int):
        self.account = replace(self.account, session_revocation_pending=False)
        return Success(self.account)


class FakeIssuer:
    def issue(
        self,
        *,
        target_account_id: str,
        target_role: str,
        target_permissions: frozenset[str],
        admin_account_id: str,
    ) -> tuple[str, int]:
        return (f"token-for-{target_account_id}-by-{admin_account_id}", 900)


@pytest.mark.asyncio
async def test_account_reads_return_auth_owned_dto_and_typed_missing_failure() -> None:
    store = FakeAccountStore()

    by_id = await auth_public.get_account_by_id(store, "account-1")
    by_email = await auth_public.get_account_by_email(store, "ACCOUNT@example.com")
    missing = await auth_public.get_account_by_id(store, "missing")

    assert isinstance(by_id, Success)
    assert by_id.unwrap().display_name == "Account One"
    assert not hasattr(by_id.unwrap(), "password_hash")
    assert isinstance(by_email, Success)
    assert by_email.unwrap().id == "account-1"
    assert isinstance(missing, Failure)
    assert missing.failure().code.value == "USER_NOT_FOUND"


@pytest.mark.asyncio
async def test_profile_mutation_success_and_typed_failure() -> None:
    store = FakeAccountStore()
    mutation = auth_public.ProfileMutation(display_name="Renamed", avatar_url="avatar", locale="fr")

    result = await auth_public.mutate_profile(store, "account-1", mutation)
    assert isinstance(result, Success)
    assert result.unwrap().display_name == "Renamed"
    assert result.unwrap().locale == "fr"

    store.failure_operation = "profile"
    failed = await auth_public.mutate_profile(store, "account-1", mutation)
    assert isinstance(failed, Failure)
    assert isinstance(failed.failure(), AuthInfrastructureError)


@pytest.mark.asyncio
async def test_password_change_success_and_wrong_current_password_failure() -> None:
    store = FakeAccountStore()
    sessions = FakeSessionPort()
    change = auth_public.PasswordChange(current_password="CurrentPass1", new_password="NewPass2")

    result = await auth_public.change_password(store, sessions, FakeHasher(), "account-1", change)
    assert isinstance(result, Success)
    assert store.replaced_hash == "hash:NewPass2"

    wrong = await auth_public.change_password(
        store,
        sessions,
        FakeHasher(),
        "account-1",
        change.model_copy(update={"current_password": "WrongPass1"}),
    )
    assert isinstance(wrong, Failure)
    assert isinstance(wrong.failure(), AuthAuthenticationError)


@pytest.mark.asyncio
async def test_single_session_revocation_success_and_typed_missing_failure() -> None:
    sessions = FakeSessionPort()

    result = await auth_public.revoke_session(sessions, "session-1", "account-1")
    missing = await auth_public.revoke_session(sessions, "session-1", "account-1")

    assert isinstance(result, Success)
    assert result.unwrap().session_ids == ("session-1",)
    assert isinstance(missing, Failure)
    assert isinstance(missing.failure(), AuthSessionNotFoundError)


@pytest.mark.asyncio
async def test_all_session_revocation_success_and_typed_failure() -> None:
    sessions = FakeSessionPort()
    result = await auth_public.revoke_all_sessions(sessions, "account-1")
    assert isinstance(result, Success)
    assert result.unwrap().session_ids == ("session-1",)

    sessions.failure = True
    failed = await auth_public.revoke_all_sessions(sessions, "account-1")
    assert isinstance(failed, Failure)
    assert isinstance(failed.failure(), AuthInfrastructureError)


@pytest.mark.asyncio
async def test_role_and_status_administration_success_and_self_failure() -> None:
    store = FakeAccountStore()
    administration = FakeAdministration(store.account)
    sessions = FakeSessionPort()

    role = await auth_public.update_account_role(
        store,
        administration,
        sessions,
        auth_public.RoleAdministration(
            account_id="account-1", role="moderator", requesting_admin_id="admin-1"
        ),
    )
    assert isinstance(role, Success)
    assert role.unwrap().role == "moderator"
    status = await auth_public.set_account_status(
        store,
        administration,
        sessions,
        auth_public.StatusAdministration(
            account_id="account-1", is_active=False, requesting_admin_id="admin-1"
        ),
    )
    assert isinstance(status, Success)
    assert status.unwrap().is_active is False

    self_change = await auth_public.set_account_status(
        store,
        administration,
        sessions,
        auth_public.StatusAdministration(
            account_id="admin-1", is_active=False, requesting_admin_id="admin-1"
        ),
    )
    assert isinstance(self_change, Failure)
    assert self_change.failure().code.value == "USER_CONFLICT"


@pytest.mark.asyncio
async def test_impersonation_success_and_disabled_account_failure() -> None:
    store = FakeAccountStore()
    result = await auth_public.issue_impersonation_token(
        store, FakeIssuer(), "account-1", "admin-1"
    )
    assert isinstance(result, Success)
    assert result.unwrap().access_token == "token-for-account-1-by-admin-1"

    store.account = replace(store.account, is_active=False)
    failed = await auth_public.issue_impersonation_token(
        store, FakeIssuer(), "account-1", "admin-1"
    )
    assert isinstance(failed, Failure)
    assert failed.failure().code.value == "FORBIDDEN"


def test_public_surface_has_no_persistence_or_transport_names() -> None:
    denylist = ("repository", "adapter", "beanie", "User", "dto.http")
    names = [name for name in dir(auth_public) if not name.startswith("__")]
    assert not [name for name in names if any(part in name for part in denylist)]
    assert set(auth_public.__all__) == {
        "AccountRead",
        "ImpersonationToken",
        "PasswordChange",
        "ProfileMutation",
        "RoleAdministration",
        "SessionRevocation",
        "StatusAdministration",
        "change_password",
        "get_account_by_email",
        "get_account_by_id",
        "issue_impersonation_token",
        "mutate_profile",
        "revoke_all_sessions",
        "revoke_session",
        "set_account_status",
        "update_account_role",
    }
