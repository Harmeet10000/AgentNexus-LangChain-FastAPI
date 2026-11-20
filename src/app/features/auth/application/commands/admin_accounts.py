"""Administrative account commands owned by the auth bounded context."""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure, Success

from app.features.auth.application.dto.public import AccountRead, ImpersonationToken
from app.features.auth.domain.errors import (
    AuthAdminAuthorizationError,
    AuthAdminConflictError,
    AuthAdminInfrastructureError,
    AuthNotFoundError,
)
from app.kernel.errors import FeatureError
from app.observability.logging import logger, trace_layer

if TYPE_CHECKING:
    from app.features.auth.application.ports.admin_accounts import AdminAccountRepositoryPort
    from app.features.auth.application.public_operations import ImpersonationIssuer, SessionPort
    from app.features.auth.domain.errors import AuthError, AuthResult
    from app.features.auth.domain.user import User, UserRole


def _admin_failure(error: AuthError) -> AuthError:
    if isinstance(
        error, AuthAdminInfrastructureError | AuthAdminConflictError | AuthAdminAuthorizationError
    ):
        return error
    kind = getattr(error, "kind", None)
    if getattr(kind, "value", kind) == "infrastructure":
        return AuthAdminInfrastructureError(
            message=error.message,
            details=error.details,
            source="auth_admin_command",
            operation="revoke_sessions",
        )
    if getattr(kind, "value", kind) == "conflict":
        return AuthAdminConflictError(
            message=error.message,
            details=error.details,
            source="auth_admin_command",
        )
    if getattr(kind, "value", kind) == "authorization":
        return AuthAdminAuthorizationError(
            message=error.message,
            details=error.details,
            source="auth_admin_command",
            operation="impersonate",
        )
    if isinstance(error, FeatureError):
        return error
    return AuthAdminInfrastructureError(
        message=str(getattr(error, "message", "Administrative account operation failed")),
        details=getattr(error, "details", None),
        source="auth_admin_command",
        operation="admin_account_operation",
    )


class AdminAccounts:
    def __init__(
        self,
        accounts: AdminAccountRepositoryPort,
        session_port: SessionPort,
        *,
        issuer: ImpersonationIssuer | None = None,
    ) -> None:
        self._accounts = accounts
        self._session_port = session_port
        self._issuer = issuer

    @trace_layer("command")
    async def update_role(
        self, user_id: str, new_role: UserRole, requesting_admin_id: str
    ) -> AuthResult[AccountRead]:
        if user_id == requesting_admin_id:
            return Failure(
                AuthAdminConflictError(
                    message="Admins cannot update their own role",
                    source="auth_admin_command",
                    operation="update_role",
                )
            )
        found = await self._accounts.find_by_id(user_id)
        if isinstance(found, Failure):
            return Failure(_admin_failure(found.failure()))
        current = found.unwrap()
        if current is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": user_id},
                    source="auth_admin_command",
                )
            )
        if current.role == new_role and not getattr(current, "session_revocation_pending", False):
            return Success(_account_read(current))
        if current.role == new_role:
            user = current
        else:
            updated = await self._accounts.update_role(current, new_role)
            if isinstance(updated, Failure):
                return Failure(_admin_failure(updated.failure()))
            user = updated.unwrap()
        revoked = await self._session_port.revoke_all_user_sessions(
            user_id=user_id,
            reason="role_changed",
            authz_revision=getattr(user, "role_revision", 0),
        )
        if isinstance(revoked, Failure):
            return Failure(_admin_failure(revoked.failure()))
        if getattr(user, "session_revocation_pending", False):
            cleared = await self._accounts.clear_session_revocation_pending(
                user, expected_revision=getattr(user, "role_revision", 0)
            )
            if isinstance(cleared, Failure):
                return Failure(_admin_failure(cleared.failure()))
            user = cleared.unwrap()
        logger.bind(target_user_id=user_id, new_role=new_role, admin_id=requesting_admin_id).info(
            "User role updated"
        )
        return Success(_account_read(user))

    @trace_layer("command")
    async def set_active(
        self, user_id: str, *, is_active: bool, requesting_admin_id: str
    ) -> AuthResult[AccountRead]:
        if user_id == requesting_admin_id:
            return Failure(
                AuthAdminConflictError(
                    message="Admins cannot deactivate themselves",
                    source="auth_admin_command",
                    operation="set_active",
                )
            )
        found = await self._accounts.find_by_id(user_id)
        if isinstance(found, Failure):
            return Failure(_admin_failure(found.failure()))
        current = found.unwrap()
        if current is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": user_id},
                    source="auth_admin_command",
                )
            )
        updated = await self._accounts.set_active(current, is_active=is_active)
        if isinstance(updated, Failure):
            return Failure(_admin_failure(updated.failure()))
        user = updated.unwrap()
        if not is_active:
            revoked = await self._session_port.revoke_all_user_sessions(
                user_id=user_id, reason="account_deactivated"
            )
            if isinstance(revoked, Failure):
                return Failure(_admin_failure(revoked.failure()))
        logger.bind(target_user_id=user_id, is_active=is_active, admin_id=requesting_admin_id).info(
            "User active status updated"
        )
        return Success(_account_read(user))

    @trace_layer("command")
    async def hard_delete(self, user_id: str, requesting_admin_id: str) -> AuthResult[None]:
        if user_id == requesting_admin_id:
            return Failure(
                AuthAdminConflictError(
                    message="Admins cannot delete themselves",
                    source="auth_admin_command",
                )
            )
        found = await self._accounts.find_by_id(user_id)
        if isinstance(found, Failure):
            return Failure(_admin_failure(found.failure()))
        user = found.unwrap()
        if user is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": user_id},
                    source="auth_admin_command",
                )
            )
        revoked = await self._session_port.revoke_all_user_sessions(
            user_id=user_id, reason="account_deleted"
        )
        if isinstance(revoked, Failure):
            return Failure(_admin_failure(revoked.failure()))
        deleted = await self._accounts.hard_delete(user)
        if isinstance(deleted, Failure):
            return Failure(_admin_failure(deleted.failure()))
        logger.bind(target_user_id=user_id, admin_id=requesting_admin_id).info("User hard deleted")
        return Success(None)

    @trace_layer("command")
    async def impersonate(
        self, target_user_id: str, admin_user_id: str
    ) -> AuthResult[ImpersonationToken]:
        if self._issuer is None:
            message = "Impersonation token issuer is not configured"
            return Failure(
                AuthAdminInfrastructureError(
                    message=message,
                    source="auth_admin_command",
                    operation="impersonate",
                )
            )
        found = await self._accounts.find_by_id(target_user_id)
        if isinstance(found, Failure):
            return Failure(_admin_failure(found.failure()))
        user = found.unwrap()
        if user is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": target_user_id},
                    source="auth_admin_command",
                )
            )
        if target_user_id == admin_user_id:
            return Failure(
                AuthAdminAuthorizationError(
                    message="Cannot impersonate yourself",
                    source="auth_admin_command",
                    operation="impersonate",
                )
            )
        if not user.is_active:
            return Failure(
                AuthAdminAuthorizationError(
                    message="Cannot impersonate a disabled account",
                    source="auth_admin_command",
                    operation="impersonate",
                )
            )
        token, expires_in = self._issuer.issue(
            target_account_id=str(user.id),
            target_role=user.role.value,
            target_permissions=frozenset(permission.value for permission in user.get_permissions()),
            admin_account_id=admin_user_id,
        )
        logger.bind(target_user_id=target_user_id, admin_id=admin_user_id).warning(
            "Admin impersonation session created"
        )
        return Success(
            ImpersonationToken(
                access_token=token,
                expires_in=expires_in,
                impersonating_account_id=admin_user_id,
            )
        )


def _account_read(user: User) -> AccountRead:
    permissions = (
        user.get_permissions()
        if hasattr(user, "get_permissions")
        else getattr(user, "permissions", ())
    )
    return AccountRead(
        id=str(user.id),
        email=user.email,
        display_name=user.full_name,
        avatar_url=getattr(user, "avatar_url", None),
        locale=getattr(user, "locale", None),
        role=user.role.value,
        is_verified=user.is_verified,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        permissions=frozenset(
            getattr(permission, "value", str(permission)) for permission in permissions
        ),
        legal_roles=tuple(
            getattr(role, "value", str(role)) for role in getattr(user, "legal_roles", ())
        ),
        role_revision=max(0, int(getattr(user, "role_revision", 0))),
        oauth_providers=tuple(account.provider for account in getattr(user, "oauth_accounts", ())),
    )


__all__ = ["AdminAccounts"]
