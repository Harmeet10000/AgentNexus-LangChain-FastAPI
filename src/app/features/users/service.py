import math

from returns.result import Failure, Success

from app.features.auth import RefreshTokenRepository, User, UserRole, create_impersonation_token
from app.utils import logger

from .dto import (
    ImpersonateResponse,
    PaginatedData,
    UserAdminResponse,
)
from .errors import (
    UsersAuthorizationError,
    UsersConflictError,
    UsersInfrastructureError,
    UsersNotFoundError,
    UsersResult,
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
        self._user_repo: UserAdminRepository = user_repo
        self._token_repo: RefreshTokenRepository = token_repo

    async def _get_user(self, user_id: str) -> UsersResult[User]:
        result = await self._user_repo.find_by_id(user_id)
        if isinstance(result, Failure):
            return result
        user = result.unwrap()
        if user is None:
            return Failure(
                UsersNotFoundError(
                    message="User not found",
                    details={"user_id": user_id},
                    source="users_service",
                    user_id=user_id,
                )
            )
        return Success(user)

    async def list_users(
        self,
        page: int,
        per_page: int,
        role: UserRole | None = None,
        is_active: bool | None = None,
        search: str | None = None,
    ) -> UsersResult[PaginatedData[UserAdminResponse]]:
        per_page = min(per_page, 100)  # hard cap — prevent unbounded queries
        result = await self._user_repo.list_users(
            page=page,
            per_page=per_page,
            role=role,
            is_active=is_active,
            search=search,
        )
        if isinstance(result, Failure):
            return result
        items, total = result.unwrap()
        return Success(
            PaginatedData(
                items=[_to_admin_response(u) for u in items],
                total=total,
                page=page,
                per_page=per_page,
                pages=math.ceil(total / per_page) if total else 0,
            )
        )

    async def get_user(self, user_id: str) -> UsersResult[UserAdminResponse]:
        result = await self._get_user(user_id)
        if isinstance(result, Failure):
            return result
        return Success(_to_admin_response(result.unwrap()))

    async def update_role(
        self,
        user_id: str,
        new_role: UserRole,
        requesting_admin_id: str,
    ) -> UsersResult[UserAdminResponse]:
        if user_id == requesting_admin_id:
            return Failure(
                UsersConflictError(
                    message="Admins cannot update their own role",
                    source="users_service",
                    operation="update_role",
                )
            )
        lookup = await self._get_user(user_id)
        if isinstance(lookup, Failure):
            return lookup
        updated = await self._user_repo.update_role(lookup.unwrap(), new_role)
        if isinstance(updated, Failure):
            return updated
        user = updated.unwrap()
        logger.bind(
            target_user_id=user_id,
            new_role=new_role,
            admin_id=requesting_admin_id,
        ).info("User role updated")
        return Success(_to_admin_response(user))

    async def set_active(
        self,
        user_id: str,
        *,
        is_active: bool,
        requesting_admin_id: str,
    ) -> UsersResult[UserAdminResponse]:
        if user_id == requesting_admin_id:
            return Failure(
                UsersConflictError(
                    message="Admins cannot deactivate themselves",
                    source="users_service",
                    operation="set_active",
                )
            )
        lookup = await self._get_user(user_id)
        if isinstance(lookup, Failure):
            return lookup
        updated = await self._user_repo.set_active(lookup.unwrap(), is_active=is_active)
        if isinstance(updated, Failure):
            return updated
        user = updated.unwrap()
        if not is_active:
            # Force all sessions offline when account is deactivated
            revoked = await self._token_repo.revoke_all_user_sessions(
                user_id=user_id,
                reason="account_deactivated",
            )
            if isinstance(revoked, Failure):
                error = revoked.failure()
                return Failure(
                    UsersInfrastructureError(
                        message=error.message,
                        details=error.details,
                        source="users_service",
                        operation="revoke_sessions",
                    )
                )
        logger.bind(
            target_user_id=user_id,
            is_active=is_active,
            admin_id=requesting_admin_id,
        ).info("User active status updated")
        return Success(_to_admin_response(user))

    async def hard_delete(
        self,
        user_id: str,
        requesting_admin_id: str,
    ) -> UsersResult[None]:
        if user_id == requesting_admin_id:
            return Failure(
                UsersConflictError(
                    message="Admins cannot delete themselves",
                    source="users_service",
                    operation="hard_delete",
                )
            )
        lookup = await self._get_user(user_id)
        if isinstance(lookup, Failure):
            return lookup
        user = lookup.unwrap()
        # Revoke sessions before deletion so Redis doesn't hold orphaned keys
        revoked = await self._token_repo.revoke_all_user_sessions(
            user_id=user_id,
            reason="account_deleted",
        )
        if isinstance(revoked, Failure):
            error = revoked.failure()
            return Failure(
                UsersInfrastructureError(
                    message=error.message,
                    details=error.details,
                    source="users_service",
                    operation="revoke_sessions",
                )
            )
        deleted = await self._user_repo.hard_delete(user)
        if isinstance(deleted, Failure):
            return deleted
        logger.bind(target_user_id=user_id, admin_id=requesting_admin_id).info("User hard deleted")
        return Success(None)

    async def impersonate(
        self,
        target_user_id: str,
        admin_user_id: str,
    ) -> UsersResult[ImpersonateResponse]:
        if target_user_id == admin_user_id:
            return Failure(
                UsersAuthorizationError(
                    message="Cannot impersonate yourself",
                    source="users_service",
                    operation="impersonate",
                )
            )
        lookup = await self._get_user(target_user_id)
        if isinstance(lookup, Failure):
            return lookup
        user = lookup.unwrap()
        if not user.is_active:
            return Failure(
                UsersAuthorizationError(
                    message="Cannot impersonate a disabled account",
                    source="users_service",
                    operation="impersonate",
                )
            )

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
        return Success(
            ImpersonateResponse(
                access_token=access_token,
                token_type="bearer",  # noqa: S106
                expires_in=expires_in,
                impersonating_user_id=admin_user_id,
            )
        )
