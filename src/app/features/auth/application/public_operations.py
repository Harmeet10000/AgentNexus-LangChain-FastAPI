"""Auth-owned application operations for the public feature seam.

The protocols in this module are deliberately smaller than the existing
MongoDB and Redis classes.  They are application ports: adapters are composed
outside this module and therefore cannot leak into the public surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from returns.result import Failure, Success

from app.features.auth.application.dto.public import (
    AccountRead,
    ImpersonationToken,
    SessionRevocation,
)
from app.features.auth.domain.errors import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthNotFoundError,
    AuthSessionNotFoundError,
    AuthValidationError,
)
from app.features.auth.domain.user import UserRole

if TYPE_CHECKING:
    from datetime import datetime

    from returns.result import Result

    from app.features.auth.application.dto.public import (
        PasswordChange,
        ProfileMutation,
        RoleAdministration,
        StatusAdministration,
    )
    from app.features.auth.domain.errors import AuthError, AuthResult


_SOURCE = "auth_public_operations"


class _Account(Protocol):
    id: str
    email: str
    full_name: str | None
    avatar_url: str | None
    is_verified: bool
    is_active: bool
    role: object
    created_at: datetime
    updated_at: datetime
    password_hash: str | None
    role_revision: int
    legal_roles: object
    permissions: object


class AccountReader(Protocol):
    async def find_by_id(self, account_id: str) -> Result[_Account | None, AuthError]: ...

    async def find_by_email(self, email: str) -> Result[_Account | None, AuthError]: ...


class ProfileWriter(Protocol):
    async def update_profile(
        self, account_id: str, *, full_name: str
    ) -> Result[object, AuthError]: ...

    async def set_avatar_url(
        self, account_id: str, avatar_url: str | None
    ) -> Result[None, AuthError]: ...

    async def set_locale(self, account_id: str, locale: str) -> Result[None, AuthError]: ...


class CredentialWriter(Protocol):
    async def replace_password_hash(
        self, account_id: str, password_hash: str
    ) -> Result[None, AuthError]: ...


class ProfileAccountPort(AccountReader, ProfileWriter, Protocol):
    """Account read/write port needed by the profile operation."""


class CredentialAccountPort(AccountReader, CredentialWriter, Protocol):
    """Account read/write port needed by the credential operation."""


class _Session(Protocol):
    session_id: str
    user_id: str
    family_id: str | None


class SessionPort(Protocol):
    async def get_session(self, session_id: str) -> Result[_Session | None, AuthError]: ...

    async def revoke_refresh_family(
        self, family_id: str, *, reason: str, user_id: str, session_id: str
    ) -> Result[None, AuthError]: ...

    async def revoke_session(
        self, session_id: str, user_id: str, reason: str = "logout"
    ) -> Result[None, AuthError]: ...

    async def revoke_all_user_sessions(
        self,
        user_id: str,
        except_session_id: str | None = None,
        reason: str = "revoke_all",
        authz_revision: int | None = None,
    ) -> Result[list[str], AuthError]: ...


class PasswordHasher(Protocol):
    def verify(self, hashed: str, plain: str) -> bool: ...

    def hash(self, password: str) -> str: ...


class AccountAdministrationPort(Protocol):
    async def update_role(self, account_id: str, role: str) -> Result[_Account, AuthError]: ...

    async def set_active(
        self, account_id: str, *, is_active: bool
    ) -> Result[_Account, AuthError]: ...

    async def clear_session_revocation_pending(
        self, account_id: str, *, expected_revision: int
    ) -> Result[_Account, AuthError]: ...


class ImpersonationIssuer(Protocol):
    def issue(
        self,
        *,
        target_account_id: str,
        target_role: str,
        target_permissions: frozenset[str],
        admin_account_id: str,
    ) -> tuple[str, int]: ...


def _failure(error: AuthError) -> Failure[AuthError]:
    return Failure(error)


def _missing_account(*, field: str, value: str) -> Failure[AuthError]:
    return _failure(
        AuthNotFoundError(
            message="User not found",
            details={field: value},
            source=_SOURCE,
        )
    )


def _role_value(role: object) -> str:
    value = getattr(role, "value", role)
    return str(value)


def _string_values(values: object) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list, set, frozenset)):
        return ()
    return tuple(sorted(_role_value(value) for value in values))


def _account_read(account: _Account) -> AccountRead:
    permissions = getattr(account, "permissions", ())
    return AccountRead(
        id=str(account.id),
        email=account.email,
        display_name=account.full_name,
        avatar_url=account.avatar_url,
        locale=getattr(account, "locale", None),
        role=_role_value(account.role),
        is_verified=account.is_verified,
        is_active=account.is_active,
        created_at=account.created_at,
        updated_at=account.updated_at,
        permissions=frozenset(_string_values(permissions)),
        legal_roles=_string_values(getattr(account, "legal_roles", ())),
        role_revision=max(0, int(getattr(account, "role_revision", 0))),
    )


async def _find_account(
    reader: AccountReader, *, account_id: str | None = None, email: str | None = None
) -> AuthResult[_Account]:
    result = (
        await reader.find_by_id(account_id)
        if account_id is not None
        else await reader.find_by_email(cast("str", email))
    )
    if isinstance(result, Failure):
        return Failure(result.failure())
    account = result.unwrap()
    if account is None:
        return _missing_account(
            field="account_id" if account_id is not None else "email",
            value=account_id if account_id is not None else cast("str", email),
        )
    return Success(account)


async def get_account_by_id(reader: AccountReader, account_id: str) -> AuthResult[AccountRead]:
    """Read an account by its auth-owned identifier."""

    result = await _find_account(reader, account_id=account_id)
    if isinstance(result, Failure):
        return Failure(result.failure())
    return Success(_account_read(result.unwrap()))


async def get_account_by_email(reader: AccountReader, email: str) -> AuthResult[AccountRead]:
    """Read an account by email without exposing its persistence document."""

    result = await _find_account(reader, email=email)
    if isinstance(result, Failure):
        return Failure(result.failure())
    return Success(_account_read(result.unwrap()))


async def mutate_profile(
    account_store: ProfileAccountPort,
    account_id: str,
    mutation: ProfileMutation,
) -> AuthResult[AccountRead]:
    """Apply profile-owned fields and return the refreshed auth read model."""

    existing = await _find_account(account_store, account_id=account_id)
    if isinstance(existing, Failure):
        return Failure(existing.failure())

    if mutation.display_name is not None:
        changed = await account_store.update_profile(account_id, full_name=mutation.display_name)
        if isinstance(changed, Failure):
            return Failure(changed.failure())
    if mutation.avatar_url is not None or "avatar_url" in mutation.model_fields_set:
        changed = await account_store.set_avatar_url(account_id, mutation.avatar_url)
        if isinstance(changed, Failure):
            return Failure(changed.failure())
    if mutation.locale is not None:
        changed = await account_store.set_locale(account_id, mutation.locale)
        if isinstance(changed, Failure):
            return Failure(changed.failure())

    return await get_account_by_id(account_store, account_id)


async def change_password(
    account_store: CredentialAccountPort,
    session_port: SessionPort,
    password_hasher: PasswordHasher,
    account_id: str,
    change: PasswordChange,
    *,
    current_session_id: str | None = None,
) -> AuthResult[None]:
    """Verify and replace a password, optionally revoking other sessions."""

    existing = await _find_account(account_store, account_id=account_id)
    if isinstance(existing, Failure):
        return Failure(existing.failure())
    account = existing.unwrap()
    if account.password_hash is None:
        return _failure(
            AuthConflictError(
                message=(
                    "Password cannot be changed on an OAuth-only account. "
                    "Link a password via account settings."
                ),
                source=_SOURCE,
            )
        )
    if not password_hasher.verify(account.password_hash, change.current_password):
        return _failure(
            AuthAuthenticationError(message="Current password is incorrect", source=_SOURCE)
        )
    if change.current_password == change.new_password:
        return _failure(
            AuthConflictError(
                message="New password must differ from current password", source=_SOURCE
            )
        )

    saved = await account_store.replace_password_hash(
        account_id, password_hasher.hash(change.new_password)
    )
    if isinstance(saved, Failure):
        return Failure(saved.failure())
    if change.revoke_other_sessions:
        revoked = await session_port.revoke_all_user_sessions(
            user_id=account_id,
            except_session_id=current_session_id,
            reason="password_change",
        )
        if isinstance(revoked, Failure):
            return Failure(revoked.failure())
    return Success(None)


async def revoke_session(
    session_port: SessionPort,
    session_id: str,
    account_id: str,
    *,
    reason: str = "manual_revoke",
) -> AuthResult[SessionRevocation]:
    """Revoke one session after proving it belongs to the addressed account."""

    found = await session_port.get_session(session_id)
    if isinstance(found, Failure):
        return Failure(found.failure())
    session = found.unwrap()
    if session is None:
        return Failure(
            AuthSessionNotFoundError(
                message="Session not found", source=_SOURCE, session_id=session_id
            )
        )
    if session.user_id != account_id:
        return _failure(
            AuthAuthorizationError(message="Cannot revoke another user's session", source=_SOURCE)
        )
    if session.family_id:
        family = await session_port.revoke_refresh_family(
            session.family_id,
            reason=reason,
            user_id=account_id,
            session_id=session_id,
        )
        if isinstance(family, Failure):
            return Failure(family.failure())
    revoked = await session_port.revoke_session(session_id, account_id, reason=reason)
    if isinstance(revoked, Failure):
        return Failure(revoked.failure())
    return Success(SessionRevocation(session_ids=(session_id,)))


async def revoke_all_sessions(
    session_port: SessionPort,
    account_id: str,
    *,
    except_session_id: str | None = None,
    reason: str = "revoke_all",
    authz_revision: int | None = None,
) -> AuthResult[SessionRevocation]:
    """Revoke all sessions for an account, optionally retaining one session."""

    revoked = await session_port.revoke_all_user_sessions(
        user_id=account_id,
        except_session_id=except_session_id,
        reason=reason,
        authz_revision=authz_revision,
    )
    if isinstance(revoked, Failure):
        return Failure(revoked.failure())
    return Success(SessionRevocation(session_ids=tuple(revoked.unwrap())))


def _validate_role(role: str) -> AuthResult[str]:
    try:
        return Success(UserRole(role).value)
    except ValueError:
        return Failure(
            AuthValidationError(message="Invalid role", details={"role": role}, source=_SOURCE)
        )


async def update_account_role(
    account_store: AccountReader,
    administration: AccountAdministrationPort,
    session_port: SessionPort,
    request: RoleAdministration,
) -> AuthResult[AccountRead]:
    """Change an account role and invalidate its authorization sessions."""

    if request.account_id == request.requesting_admin_id:
        return _failure(
            AuthConflictError(message="Admins cannot update their own role", source=_SOURCE)
        )
    role = _validate_role(request.role)
    if isinstance(role, Failure):
        return Failure(role.failure())

    existing = await _find_account(account_store, account_id=request.account_id)
    if isinstance(existing, Failure):
        return Failure(existing.failure())
    current = existing.unwrap()
    current_role = _role_value(current.role)
    pending = bool(getattr(current, "session_revocation_pending", False))
    if current_role == role.unwrap() and not pending:
        return Success(_account_read(current))

    if current_role == role.unwrap():
        updated = Success(current)
    else:
        updated = await administration.update_role(request.account_id, role.unwrap())
    if isinstance(updated, Failure):
        return Failure(updated.failure())
    account = updated.unwrap()
    revoked = await revoke_all_sessions(
        session_port,
        request.account_id,
        reason="role_changed",
        authz_revision=int(getattr(account, "role_revision", 0)),
    )
    if isinstance(revoked, Failure):
        return Failure(revoked.failure())
    if bool(getattr(account, "session_revocation_pending", False)):
        cleared = await administration.clear_session_revocation_pending(
            request.account_id,
            expected_revision=int(getattr(account, "role_revision", 0)),
        )
        if isinstance(cleared, Failure):
            return Failure(cleared.failure())
        account = cleared.unwrap()
    return Success(_account_read(account))


async def set_account_status(
    account_store: AccountReader,
    administration: AccountAdministrationPort,
    session_port: SessionPort,
    request: StatusAdministration,
) -> AuthResult[AccountRead]:
    """Change activation status and force deactivated accounts offline."""

    if request.account_id == request.requesting_admin_id:
        return _failure(
            AuthConflictError(message="Admins cannot deactivate themselves", source=_SOURCE)
        )
    existing = await _find_account(account_store, account_id=request.account_id)
    if isinstance(existing, Failure):
        return Failure(existing.failure())
    updated = await administration.set_active(request.account_id, is_active=request.is_active)
    if isinstance(updated, Failure):
        return Failure(updated.failure())
    account = updated.unwrap()
    if not request.is_active:
        revoked = await revoke_all_sessions(
            session_port, request.account_id, reason="account_deactivated"
        )
        if isinstance(revoked, Failure):
            return Failure(revoked.failure())
    return Success(_account_read(account))


async def issue_impersonation_token(
    reader: AccountReader,
    issuer: ImpersonationIssuer,
    target_account_id: str,
    admin_account_id: str,
) -> AuthResult[ImpersonationToken]:
    """Issue a short-lived token for an active account on an admin's behalf."""

    if target_account_id == admin_account_id:
        return _failure(
            AuthAuthorizationError(message="Cannot impersonate yourself", source=_SOURCE)
        )
    found = await _find_account(reader, account_id=target_account_id)
    if isinstance(found, Failure):
        return Failure(found.failure())
    account = found.unwrap()
    if not account.is_active:
        return _failure(
            AuthAuthorizationError(message="Cannot impersonate a disabled account", source=_SOURCE)
        )
    read = _account_read(account)
    token, expires_in = issuer.issue(
        target_account_id=read.id,
        target_role=read.role,
        target_permissions=read.permissions,
        admin_account_id=admin_account_id,
    )
    return Success(
        ImpersonationToken(
            access_token=token,
            expires_in=expires_in,
            impersonating_account_id=admin_account_id,
        )
    )


__all__ = [
    "change_password",
    "get_account_by_email",
    "get_account_by_id",
    "issue_impersonation_token",
    "mutate_profile",
    "revoke_all_sessions",
    "revoke_session",
    "set_account_status",
    "update_account_role",
]
