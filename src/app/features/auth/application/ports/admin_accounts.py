"""Application ports for the auth administrative account projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from returns.result import Result

    from app.features.auth.domain.errors import AuthError
    from app.features.auth.domain.user import User, UserRole


class AdminAccountReader(Protocol):
    async def find_by_id(self, user_id: str) -> Result[User | None, AuthError]: ...

    async def list_users(
        self,
        *,
        page: int,
        per_page: int,
        role: UserRole | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> Result[tuple[list[User], int], AuthError]: ...


class AdminAccountWriter(Protocol):
    async def update_role(self, user: User, role: UserRole) -> Result[User, AuthError]: ...

    async def set_active(self, user: User, *, is_active: bool) -> Result[User, AuthError]: ...

    async def clear_session_revocation_pending(
        self, user: User, *, expected_revision: int | None = None
    ) -> Result[User, AuthError]: ...

    async def hard_delete(self, user: User) -> Result[None, AuthError]: ...


class AdminAccountRepositoryPort(AdminAccountReader, AdminAccountWriter, Protocol):
    """The complete repository boundary used by admin queries and commands."""
