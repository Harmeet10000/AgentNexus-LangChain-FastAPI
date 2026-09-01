from datetime import UTC, datetime

from beanie import PydanticObjectId
from beanie.operators import Or, RegEx, Set
from pymongo.errors import PyMongoError
from returns.result import Failure, Success

from app.features.auth import User, UserRole
from app.utils import logger

from .errors import UsersInfrastructureError, UsersResult, UsersValidationError


class UserAdminRepository:
    """Admin-scoped user queries.

    Directly uses the Beanie User document. No Motor db instance needed —
    Beanie manages the connection at the document class level after init.
    """

    @staticmethod
    async def find_by_id(user_id: str) -> UsersResult[User | None]:
        if not PydanticObjectId.is_valid(user_id):
            return Failure(
                UsersValidationError(
                    message="Invalid user identifier",
                    details={"user_id": user_id},
                    source="users_repository",
                )
            )
        try:
            return Success(await User.get(PydanticObjectId(user_id)))
        except PyMongoError as exc:
            logger.bind(operation="find_by_id", user_id=user_id).error(
                "users_repository_failed", error=str(exc)
            )
            return Failure(
                UsersInfrastructureError(
                    message="User lookup failed",
                    details={"user_id": user_id, "error": str(exc)},
                    source="users_repository",
                    operation="find_by_id",
                )
            )

    @staticmethod
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
            logger.bind(operation="list_users").error("users_repository_failed", error=str(exc))
            return Failure(
                UsersInfrastructureError(
                    message="User listing failed",
                    details={"error": str(exc)},
                    source="users_repository",
                    operation="list_users",
                )
            )

    @staticmethod
    async def update_role(user: User, role: UserRole) -> UsersResult[User]:
        try:
            await user.update(Set({User.role: role, User.updated_at: datetime.now(UTC)}))
            user.role = role
            return Success(user)
        except PyMongoError as exc:
            logger.bind(operation="update_role", user_id=str(user.id)).error(
                "users_repository_failed", error=str(exc)
            )
            return Failure(
                UsersInfrastructureError(
                    message="User role update failed",
                    details={"user_id": str(user.id), "error": str(exc)},
                    source="users_repository",
                    operation="update_role",
                )
            )

    @staticmethod
    async def set_active(user: User, *, is_active: bool) -> UsersResult[User]:
        try:
            await user.update(Set({User.is_active: is_active, User.updated_at: datetime.now(UTC)}))
            user.is_active = is_active
            return Success(user)
        except PyMongoError as exc:
            logger.bind(operation="set_active", user_id=str(user.id)).error(
                "users_repository_failed", error=str(exc)
            )
            return Failure(
                UsersInfrastructureError(
                    message="User active state update failed",
                    details={"user_id": str(user.id), "error": str(exc)},
                    source="users_repository",
                    operation="set_active",
                )
            )

    @staticmethod
    async def hard_delete(user: User) -> UsersResult[None]:
        try:
            await user.delete()
            return Success(None)
        except PyMongoError as exc:
            logger.bind(operation="hard_delete", user_id=str(user.id)).error(
                "users_repository_failed", error=str(exc)
            )
            return Failure(
                UsersInfrastructureError(
                    message="User deletion failed",
                    details={"user_id": str(user.id), "error": str(exc)},
                    source="users_repository",
                    operation="hard_delete",
                )
            )
