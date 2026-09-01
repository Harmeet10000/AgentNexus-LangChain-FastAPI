from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response, status

from app.features.auth import Permission, TokenClaims, UserRole, require_permission, require_role
from app.shared.result import render_result
from app.utils import APIResponse

from .dependencies import UserAdminServiceDep
from .dto import (
    ImpersonateResponse,
    PaginatedData,
    UpdateUserRoleRequest,
    UserAdminResponse,
)

router = APIRouter(prefix="/users", tags=["users"])

# Shared RBAC guard — all routes in this router require USERS_READ at minimum.
# Individual routes layer on stricter permissions where needed.
_require_users_read = Depends(require_permission(Permission.USERS_READ))


@router.get(
    "/",
    dependencies=[_require_users_read],
)
async def list_users(
    service: UserAdminServiceDep,
    response: Response,
    *,
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(ge=1, le=100)] = 20,
    role: Annotated[UserRole | None, Query()] = None,
    is_active: Annotated[bool | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=100)] = None,
) -> APIResponse[PaginatedData[UserAdminResponse]]:
    result = await service.list_users(
        page=page,
        per_page=per_page,
        role=role,
        is_active=is_active,
        search=search,
    )
    return render_result(result, response, message="Users retrieved")


@router.get(
    "/{user_id}",
    dependencies=[_require_users_read],
)
async def get_user(
    user_id: Annotated[str, Path()],
    service: UserAdminServiceDep,
    response: Response,
) -> APIResponse[UserAdminResponse]:
    result = await service.get_user(user_id)
    return render_result(result, response, message="User retrieved")


@router.patch(
    "/{user_id}/role",
    dependencies=[Depends(require_permission(Permission.USERS_WRITE))],
)
async def update_user_role(
    user_id: Annotated[str, Path()],
    body: UpdateUserRoleRequest,
    service: UserAdminServiceDep,
    claims: Annotated[TokenClaims, Depends(require_permission(Permission.USERS_WRITE))],
    response: Response,
) -> APIResponse[UserAdminResponse]:
    result = await service.update_role(
        user_id=user_id,
        new_role=body.role,
        requesting_admin_id=claims.sub,
    )
    return render_result(result, response, message="User role updated")


@router.patch(
    "/{user_id}/activate",
    dependencies=[Depends(require_permission(Permission.USERS_WRITE))],
)
async def activate_user(
    user_id: Annotated[str, Path()],
    service: UserAdminServiceDep,
    claims: Annotated[TokenClaims, Depends(require_permission(Permission.USERS_WRITE))],
    response: Response,
) -> APIResponse[UserAdminResponse]:
    result = await service.set_active(
        user_id=user_id,
        is_active=True,
        requesting_admin_id=claims.sub,
    )
    return render_result(result, response, message="User activated")


@router.patch(
    "/{user_id}/deactivate",
    dependencies=[Depends(require_permission(Permission.USERS_WRITE))],
)
async def deactivate_user(
    user_id: Annotated[str, Path()],
    service: UserAdminServiceDep,
    claims: Annotated[TokenClaims, Depends(require_permission(Permission.USERS_WRITE))],
    response: Response,
) -> APIResponse[UserAdminResponse]:
    result = await service.set_active(
        user_id=user_id,
        is_active=False,
        requesting_admin_id=claims.sub,
    )
    return render_result(result, response, message="User deactivated")


@router.delete(
    "/{user_id}",
    response_model=APIResponse[None],
    dependencies=[Depends(require_permission(Permission.USERS_DELETE))],
)
async def delete_user(
    user_id: Annotated[str, Path()],
    service: UserAdminServiceDep,
    claims: Annotated[TokenClaims, Depends(require_permission(Permission.USERS_DELETE))],
    response: Response,
) -> APIResponse[None]:
    result = await service.hard_delete(
        user_id=user_id,
        requesting_admin_id=claims.sub,
    )
    return render_result(
        result, response, message="User permanently deleted", success_status=status.HTTP_200_OK
    )


@router.post(
    "/{user_id}/impersonate",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def impersonate_user(
    user_id: Annotated[str, Path()],
    service: UserAdminServiceDep,
    claims: Annotated[TokenClaims, Depends(require_role(UserRole.ADMIN))],
    response: Response,
) -> APIResponse[ImpersonateResponse]:
    result = await service.impersonate(
        target_user_id=user_id,
        admin_user_id=claims.sub,
    )
    return render_result(result, response, message="Impersonation token issued")
