from datetime import UTC, datetime

from beanie import PydanticObjectId
from beanie.operators import Or, RegEx, Set
from returns.result import Failure, Success

from app.features.auth import User, UserRole
from app.shared.result import AppResult, ValidationAppError


class UserAdminRepository:
    """Admin-scoped user queries.

    Directly uses the Beanie User document. No Motor db instance needed —
    Beanie manages the connection at the document class level after init.
    """

    async def find_by_id(self, user_id: str) -> AppResult[User | None]:
        if not PydanticObjectId.is_valid(user_id):
            return Failure(
                ValidationAppError(
                    code="INVALID_USER_ID",
                    message="Invalid user identifier",
                    details={"user_id": user_id},
                    source="users_repository",
                )
            )
        return Success(await User.get(PydanticObjectId(user_id)))

    async def list_users(
        self,
        page: int,
        per_page: int,
        role: UserRole | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> tuple[list[User], int]:
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

        total = await query.count()
        skip = (page - 1) * per_page
        items = await query.skip(skip).limit(per_page).to_list()
        return items, total

    async def update_role(self, user: User, role: UserRole) -> User:
        await user.update(Set({User.role: role, User.updated_at: datetime.now(UTC)}))
        user.role: UserRole = role
        return user

    async def set_active(self, user: User, *, is_active: bool) -> User:
        await user.update(Set({User.is_active: is_active, User.updated_at: datetime.now(UTC)}))
        user.is_active: bool = is_active
        return user

    async def hard_delete(self, user: User) -> None:
        await user.delete()
