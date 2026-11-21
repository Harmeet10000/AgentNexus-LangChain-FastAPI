"""MongoDB adapter for auth's administrative account projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from beanie.operators import Or, RegEx
from pymongo.errors import PyMongoError
from returns.result import Failure, Success

from app.features.auth.adapters.persistence.mongodb.identity import (
    clear_session_revocation_pending,
    delete_user_document,
    get_user_document,
    set_user_active,
    set_user_role,
)
from app.features.auth.adapters.persistence.mongodb.models import (
    UserDocument,
    to_domain_user,
)
from app.features.auth.application.ports.admin_accounts import AdminAccountRepositoryPort
from app.features.auth.domain.errors import (
    AuthAdminAuthorizationError,
    AuthAdminConflictError,
    AuthAdminInfrastructureError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthInfrastructureError,
)
from app.observability.logging import logger, trace_layer

if TYPE_CHECKING:
    from app.features.auth.domain.errors import AuthError, AuthResult
    from app.features.auth.domain.user import User, UserRole


def _translate_identity_error(error: AuthError, *, operation: str, user_id: str) -> AuthError:
    details = {"user_id": user_id, **(error.details or {})}
    if isinstance(error, AuthInfrastructureError):
        return AuthAdminInfrastructureError(
            message=error.message,
            details={**details, "error": error.message},
            source="auth_admin_repository",
            operation=operation,
        )
    if isinstance(error, AuthConflictError):
        return AuthAdminConflictError(
            message=error.message,
            details=details,
            source="auth_admin_repository",
            operation=operation,
        )
    if isinstance(error, AuthAuthorizationError):
        return AuthAdminAuthorizationError(
            message=error.message,
            details=details,
            source="auth_admin_repository",
            operation=operation,
        )
    return error.model_copy(update={"details": details, "source": "auth_admin_repository"})


def _translate[Value](
    result: AuthResult[Value], *, operation: str, user_id: str
) -> AuthResult[Value]:
    if isinstance(result, Failure):
        return Failure(
            _translate_identity_error(result.failure(), operation=operation, user_id=user_id)
        )
    return result


class AdminAccountRepository(AdminAccountRepositoryPort):
    """Auth-owned persistence adapter for administrator account operations."""

    @staticmethod
    @override
    @trace_layer("repository")
    async def find_by_id(user_id: str) -> AuthResult[User | None]:
        return _translate(await get_user_document(user_id), operation="find_by_id", user_id=user_id)

    @staticmethod
    @override
    @trace_layer("repository")
    async def list_users(
        *,
        page: int,
        per_page: int,
        role: UserRole | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> AuthResult[tuple[list[User], int]]:
        query = UserDocument.find()
        if role is not None:
            query = query.find(UserDocument.role == role)
        if is_active is not None:
            query = query.find(UserDocument.is_active == is_active)
        if search:
            query = query.find(
                Or(
                    RegEx(UserDocument.email, search, "i"),
                    RegEx(UserDocument.full_name, search, "i"),
                )
            )

        try:
            total = await query.count()
            skip = (page - 1) * per_page
            documents = await query.skip(skip).limit(per_page).to_list()
            return Success(([to_domain_user(document) for document in documents], total))
        except PyMongoError as exc:
            exc.add_note("auth_admin_repository operation=list_users")
            logger.bind(operation="list_users", error=str(exc)).exception(
                "auth_admin_repository_failed"
            )
            return Failure(
                AuthAdminInfrastructureError(
                    message="User listing failed",
                    details={"error": str(exc)},
                    source="auth_admin_repository",
                    operation="list_users",
                )
            )

    @staticmethod
    @override
    @trace_layer("repository")
    async def update_role(user: User, role: UserRole) -> AuthResult[User]:
        return _translate(
            await set_user_role(user, role), operation="update_role", user_id=str(user.id)
        )

    @staticmethod
    @override
    @trace_layer("repository")
    async def clear_session_revocation_pending(
        user: User, *, expected_revision: int | None = None
    ) -> AuthResult[User]:
        return _translate(
            await clear_session_revocation_pending(user, expected_revision=expected_revision),
            operation="clear_session_revocation_pending",
            user_id=str(user.id),
        )

    @staticmethod
    @override
    @trace_layer("repository")
    async def set_active(user: User, *, is_active: bool) -> AuthResult[User]:
        return _translate(
            await set_user_active(user, is_active=is_active),
            operation="set_active",
            user_id=str(user.id),
        )

    @staticmethod
    @override
    @trace_layer("repository")
    async def hard_delete(user: User) -> AuthResult[None]:
        return _translate(
            await delete_user_document(user), operation="hard_delete", user_id=str(user.id)
        )


__all__ = ["AdminAccountRepository"]
