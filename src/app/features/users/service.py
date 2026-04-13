import math

from app.features.auth import RefreshTokenRepository, User, UserRole, create_impersonation_token
from app.utils import ConflictException, ForbiddenException, NotFoundException, logger

from .dto import (
    ImpersonateResponse,
    PaginatedData,
    UserAdminResponse,
)
from .repository import UserAdminRepository


def _to_admin_response(user: User) -> UserAdminResponse:
    return UserAdminResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        is_verified=user.is_verified,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        oauth_providers=[acc.provider for acc in user.oauth_accounts],
    )


class UserAdminService:
    def __init__(
        self,
        user_repo: UserAdminRepository,
        token_repo: RefreshTokenRepository,
    ) -> None:
        self._user_repo = user_repo
        self._token_repo = token_repo

    async def list_users(
        self,
        page: int,
        per_page: int,
        role: UserRole | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> PaginatedData[UserAdminResponse]:
        per_page = min(per_page, 100)  # hard cap — prevent unbounded queries
        items, total = await self._user_repo.list_users(
            page=page,
            per_page=per_page,
            role=role,
            is_active=is_active,
            search=search,
        )
        return PaginatedData(
            items=[_to_admin_response(u) for u in items],
            total=total,
            page=page,
            per_page=per_page,
            pages=math.ceil(total / per_page) if total else 0,
        )

    async def get_user(self, user_id: str) -> UserAdminResponse:
        user = await self._user_repo.find_by_id(user_id)
        if user is None:
            raise NotFoundException(f"User {user_id} not found")
        return _to_admin_response(user)

    async def update_role(
        self,
        user_id: str,
        new_role: UserRole,
        requesting_admin_id: str,
    ) -> UserAdminResponse:
        if user_id == requesting_admin_id:
            raise ConflictException("Admins cannot update their own role")
        user = await self._user_repo.find_by_id(user_id)
        if user is None:
            raise NotFoundException(f"User {user_id} not found")
        user = await self._user_repo.update_role(user, new_role)
        logger.bind(
            target_user_id=user_id,
            new_role=new_role,
            admin_id=requesting_admin_id,
        ).info("User role updated")
        return _to_admin_response(user)

    async def set_active(
        self,
        user_id: str,
        *,
        is_active: bool,
        requesting_admin_id: str,
    ) -> UserAdminResponse:
        if user_id == requesting_admin_id:
            raise ConflictException("Admins cannot deactivate themselves")
        user = await self._user_repo.find_by_id(user_id)
        if user is None:
            raise NotFoundException(f"User {user_id} not found")
        user = await self._user_repo.set_active(user, is_active=is_active)
        if not is_active:
            # Force all sessions offline when account is deactivated
            await self._token_repo.revoke_all_user_sessions(
                user_id=user_id,
                reason="account_deactivated",
            )
        logger.bind(
            target_user_id=user_id,
            is_active=is_active,
            admin_id=requesting_admin_id,
        ).info("User active status updated")
        return _to_admin_response(user)

    async def hard_delete(
        self,
        user_id: str,
        requesting_admin_id: str,
    ) -> None:
        if user_id == requesting_admin_id:
            raise ConflictException("Admins cannot delete themselves")
        user = await self._user_repo.find_by_id(user_id)
        if user is None:
            raise NotFoundException(f"User {user_id} not found")
        # Revoke sessions before deletion so Redis doesn't hold orphaned keys
        await self._token_repo.revoke_all_user_sessions(
            user_id=user_id,
            reason="account_deleted",
        )
        await self._user_repo.hard_delete(user)
        logger.bind(target_user_id=user_id, admin_id=requesting_admin_id).info("User hard deleted")

    async def impersonate(
        self,
        target_user_id: str,
        admin_user_id: str,
    ) -> ImpersonateResponse:
        if target_user_id == admin_user_id:
            raise ForbiddenException("Cannot impersonate yourself")
        user = await self._user_repo.find_by_id(target_user_id)
        if user is None:
            raise NotFoundException(f"User {target_user_id} not found")
        if not user.is_active:
            raise ForbiddenException("Cannot impersonate a disabled account")

        access_token, expires_in = create_impersonation_token(
            target_user_id=str(user.id),
            target_role=user.role,
            target_permissions=user.get_permissions(),
            admin_user_id=admin_user_id,
        )
        logger.bind(
            target_user_id=target_user_id,
            admin_id=admin_user_id,
        ).warning("Admin impersonation session created")  # warning level — always audit this
        return ImpersonateResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_in,
            impersonating_user_id=admin_user_id,
        )
