from datetime import datetime

from beanie import PydanticObjectId
from beanie.operators import Or, RegEx, Set

from app.features.auth.model import User, UserRole


class UserAdminRepository:
    """Admin-scoped user queries.

    Directly uses the Beanie User document. No Motor db instance needed —
    Beanie manages the connection at the document class level after init.
    """

    async def find_by_id(self, user_id: str) -> User | None:
        try:
            return await User.get(PydanticObjectId(user_id))
        except Exception:
            return None

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
        await user.update(Set({User.role: role, User.updated_at: datetime.utcnow()}))
        user.role = role
        return user

    async def set_active(self, user: User, *, is_active: bool) -> User:
        await user.update(Set({User.is_active: is_active, User.updated_at: datetime.utcnow()}))
        user.is_active = is_active
        return user

    async def hard_delete(self, user: User) -> None:
        await user.delete()
