"""Administrative account queries owned by the auth bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from returns.result import Failure, Success

from app.features.auth.domain.errors import AuthNotFoundError

if TYPE_CHECKING:
    from app.features.auth.application.ports.admin_accounts import AdminAccountReader
    from app.features.auth.domain.errors import AuthResult
    from app.features.auth.domain.user import User, UserRole


@dataclass(frozen=True, slots=True)
class AdminUsersPage:
    items: tuple[User, ...]
    total: int
    page: int
    per_page: int

    @property
    def pages(self) -> int:
        return (self.total + self.per_page - 1) // self.per_page if self.total else 0


class AdminUsersQuery:
    def __init__(self, accounts: AdminAccountReader) -> None:
        self._accounts = accounts

    async def list_users(
        self,
        *,
        page: int,
        per_page: int,
        role: UserRole | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> AuthResult[AdminUsersPage]:
        per_page = min(per_page, 100)
        result = await self._accounts.list_users(
            page=page,
            per_page=per_page,
            role=role,
            is_active=is_active,
            search=search,
        )
        if isinstance(result, Failure):
            return Failure(result.failure())
        items, total = result.unwrap()
        return Success(
            AdminUsersPage(items=tuple(items), total=total, page=page, per_page=per_page)
        )

    async def get_user(self, user_id: str) -> AuthResult[User]:
        result = await self._accounts.find_by_id(user_id)
        if isinstance(result, Failure):
            return Failure(result.failure())
        user = result.unwrap()
        if user is None:
            return Failure(
                AuthNotFoundError(
                    message="User not found",
                    details={"user_id": user_id},
                    source="auth_admin_query",
                )
            )
        return Success(user)


__all__ = ["AdminUsersPage", "AdminUsersQuery"]
