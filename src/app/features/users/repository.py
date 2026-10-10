from typing import assert_never

from beanie.operators import Or, RegEx
from pymongo.errors import PyMongoError
from returns.result import Failure, Success

from app.features.auth.exceptions import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthValidationError,
)
from app.features.auth.identity import (
    delete_user_document,
    get_user_document,
    set_user_active,
    set_user_role,
)
from app.features.auth.models import User, UserRole
from app.observability.logging import logger, trace_layer

from .exceptions import (
    UsersAuthorizationError,
    UsersConflictError,
    UsersError,
    UsersInfrastructureError,
    UsersNotFoundError,
    UsersResult,
    UsersValidationError,
)


def _translate_identity_error(error: AuthError, *, operation: str, user_id: str) -> UsersError:
    """Map an auth-identity failure into the users error union (closed with assert_never)."""
    details = {"user_id": user_id, **(error.details or {})}
    match error:
        case AuthValidationError():
            return UsersValidationError(
                message=error.message, details=details, source="users_repository"
            )
        case AuthNotFoundError():
            return UsersNotFoundError(
                message=error.message, details=details, source="users_repository", user_id=user_id
            )
        case AuthConflictError():
            return UsersConflictError(
                message=error.message,
                details=details,
                source="users_repository",
                operation=operation,
            )
        case AuthAuthenticationError() | AuthAuthorizationError():
            return UsersAuthorizationError(
                message=error.message,
                details=details,
                source="users_repository",
                operation=operation,
            )
        case AuthInfrastructureError():
            return UsersInfrastructureError(
                message=error.message,
                details={**details, "error": error.message},
                source="users_repository",
                operation=operation,
            )
        case _:
            assert_never(error)


class UserAdminRepository:
    """Admin-scoped user queries.

    Raw document reads/writes go through auth's identity contract
    (`app.features.auth.identity`); failures are translated into the users
    error union here. The admin listing query stays local to this use case.
    """

    @staticmethod
    @trace_layer("repository")
    async def find_by_id(user_id: str) -> UsersResult[User | None]:
        result = await get_user_document(user_id)
        if isinstance(result, Failure):
            return Failure(
                _translate_identity_error(result.failure(), operation="find_by_id", user_id=user_id)
            )
        return Success(result.unwrap())

    @staticmethod
    @trace_layer("repository")
    async def list_users(
        page: int,
        per_page: int,
        role: UserRole | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> UsersResult[tuple[list[User], int]]:
        """Return (items, total_count) for the requested page."""
        query = User.find()

        if role is not None:
            query = query.find(User.role == role)
        if is_active is not None:
            query = query.find(User.is_active == is_active)
        if search:
            # Regex on email + full_name — acceptable for admin use; add text index for scale
            query = query.find(
                Or(
                    RegEx(User.email, search, "i"),
                    RegEx(User.full_name, search, "i"),
                )
            )

        try:
            total = await query.count()
            skip = (page - 1) * per_page
            items = await query.skip(skip).limit(per_page).to_list()
            return Success((items, total))
        except PyMongoError as exc:
            logger.bind(operation="list_users", error=str(exc)).exception("users_repository_failed")
            return Failure(
                UsersInfrastructureError(
                    message="User listing failed",
                    details={"error": str(exc)},
                    source="users_repository",
                    operation="list_users",
                )
            )

    @staticmethod
    @trace_layer("repository")
    async def update_role(user: User, role: UserRole) -> UsersResult[User]:
        result = await set_user_role(user, role)
        if isinstance(result, Failure):
            return Failure(
                _translate_identity_error(
                    result.failure(), operation="update_role", user_id=str(user.id)
                )
            )
        return Success(result.unwrap())

    @staticmethod
    @trace_layer("repository")
    async def set_active(user: User, *, is_active: bool) -> UsersResult[User]:
        result = await set_user_active(user, is_active=is_active)
        if isinstance(result, Failure):
            return Failure(
                _translate_identity_error(
                    result.failure(), operation="set_active", user_id=str(user.id)
                )
            )
        return Success(result.unwrap())

    @staticmethod
    @trace_layer("repository")
    async def hard_delete(user: User) -> UsersResult[None]:
        result = await delete_user_document(user)
        if isinstance(result, Failure):
            return Failure(
                _translate_identity_error(
                    result.failure(), operation="hard_delete", user_id=str(user.id)
                )
            )
        return Success(None)
