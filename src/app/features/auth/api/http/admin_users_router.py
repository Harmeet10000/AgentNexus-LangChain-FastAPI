from __future__ import annotations

from typing import (  # noqa: TC003 — FastAPI resolves annotations at runtime.
    TYPE_CHECKING,
    Annotated,
)

from fastapi import APIRouter, Depends, Path, Query, Response, status
from returns.result import Failure, Success

from app.features.auth.api.http.admin_users_dependencies import (  # noqa: TC001
    AdminAccountsDep,
    AdminUsersQueryDep,
)
from app.features.auth.api.http.admin_users_dto import (
    ImpersonateResponse,
    PaginatedData,
    UpdateUserRoleRequest,  # noqa: TC001 — FastAPI resolves the request body at runtime.
    UserAdminResponse,
)
from app.features.auth.api.http.dependencies import (
    AuthorizationContext,  # noqa: TC001 — FastAPI resolves dependency annotations at runtime.
    require_permission,
    require_role,
)
from app.features.auth.domain.user import Permission, User, UserRole
from app.shared.rate_limit import get_rate_limiter
from app.shared.result.render import APIResponse, render_result

if TYPE_CHECKING:
    from collections.abc import Callable

    from returns.result import Result

    from app.features.auth.application.dto.public import AccountRead, ImpersonationToken
    from app.features.auth.application.queries.admin_users import AdminUsersPage
    from app.features.auth.domain.errors import AuthError

router = APIRouter(prefix="/users", tags=["users"])

_require_users_read = Depends(require_permission(Permission.USERS_READ))
limit_user_mutation = get_rate_limiter(burst=30, rate=30, period=60, scope="user-mutation")


def _user_response(user: User) -> UserAdminResponse:
    return UserAdminResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        is_verified=user.is_verified,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        oauth_providers=[account.provider for account in user.oauth_accounts],
    )


def _account_response(account: AccountRead) -> UserAdminResponse:
    return UserAdminResponse(
        id=account.id,
        email=account.email,
        full_name=account.display_name,
        role=account.role,
        is_verified=account.is_verified,
        is_active=account.is_active,
        created_at=account.created_at,
        updated_at=account.updated_at,
        oauth_providers=list(account.oauth_providers),
    )


def _response(value: User | AccountRead) -> UserAdminResponse:
    return _user_response(value) if isinstance(value, User) else _account_response(value)


def _map_result[Input, Output](
    result: Result[Input, AuthError], mapper: Callable[[Input], Output]
) -> Result[Output, AuthError]:
    if isinstance(result, Failure):
        return result
    return Success(mapper(result.unwrap()))


def _page_response(page_data: AdminUsersPage) -> PaginatedData[UserAdminResponse]:
    return PaginatedData(
        items=[_user_response(user) for user in page_data.items],
        total=page_data.total,
        page=page_data.page,
        per_page=page_data.per_page,
        pages=page_data.pages,
    )


def _impersonate_response(token: ImpersonationToken) -> ImpersonateResponse:
    return ImpersonateResponse(
        access_token=token.access_token,
        token_type=token.token_type,
        expires_in=token.expires_in,
        impersonating_user_id=token.impersonating_account_id,
    )


@router.get("/", dependencies=[_require_users_read])
async def list_users(
    query: AdminUsersQueryDep,
    response: Response,
    *,
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(ge=1, le=100)] = 20,
    role: Annotated[UserRole | None, Query()] = None,
    is_active: Annotated[bool | None, Query()] = None,
    search: Annotated[str | None, Query(max_length=100)] = None,
) -> APIResponse[PaginatedData[UserAdminResponse]]:
    result = await query.list_users(
        page=page,
        per_page=per_page,
        role=role,
        is_active=is_active,
        search=search,
    )
    mapped = _map_result(result, _page_response)
    return render_result(mapped, response, message="Users retrieved")


@router.get("/{user_id}", dependencies=[_require_users_read])
async def get_user(
    user_id: Annotated[str, Path()],
    query: AdminUsersQueryDep,
    response: Response,
) -> APIResponse[UserAdminResponse]:
    result = await query.get_user(user_id)
    return render_result(_map_result(result, _user_response), response, message="User retrieved")


@router.patch("/{user_id}/role", dependencies=[Depends(limit_user_mutation)])
async def update_user_role(
    user_id: Annotated[str, Path()],
    body: UpdateUserRoleRequest,
    accounts: AdminAccountsDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_permission(Permission.USERS_WRITE)),
    ],
    response: Response,
) -> APIResponse[UserAdminResponse]:
    result = await accounts.update_role(user_id, body.role, authorization.user_id)
    return render_result(
        _map_result(result, _account_response), response, message="User role updated"
    )


@router.patch("/{user_id}/activate", dependencies=[Depends(limit_user_mutation)])
async def activate_user(
    user_id: Annotated[str, Path()],
    accounts: AdminAccountsDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_permission(Permission.USERS_WRITE)),
    ],
    response: Response,
) -> APIResponse[UserAdminResponse]:
    result = await accounts.set_active(
        user_id, is_active=True, requesting_admin_id=authorization.user_id
    )
    return render_result(_map_result(result, _account_response), response, message="User activated")


@router.patch("/{user_id}/deactivate", dependencies=[Depends(limit_user_mutation)])
async def deactivate_user(
    user_id: Annotated[str, Path()],
    accounts: AdminAccountsDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_permission(Permission.USERS_WRITE)),
    ],
    response: Response,
) -> APIResponse[UserAdminResponse]:
    result = await accounts.set_active(
        user_id, is_active=False, requesting_admin_id=authorization.user_id
    )
    return render_result(
        _map_result(result, _account_response), response, message="User deactivated"
    )


@router.delete(
    "/{user_id}",
    response_model=APIResponse[None],
    dependencies=[Depends(limit_user_mutation)],
)
async def delete_user(
    user_id: Annotated[str, Path()],
    accounts: AdminAccountsDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_permission(Permission.USERS_DELETE)),
    ],
    response: Response,
) -> APIResponse[None]:
    result = await accounts.hard_delete(user_id, authorization.user_id)
    return render_result(
        result, response, message="User permanently deleted", success_status=status.HTTP_200_OK
    )


@router.post("/{user_id}/impersonate", dependencies=[Depends(limit_user_mutation)])
async def impersonate_user(
    user_id: Annotated[str, Path()],
    accounts: AdminAccountsDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_role(UserRole.ADMIN)),
    ],
    response: Response,
) -> APIResponse[ImpersonateResponse]:
    result = await accounts.impersonate(user_id, authorization.user_id)
    mapped = _map_result(result, _impersonate_response)
    return render_result(mapped, response, message="Impersonation token issued")


__all__ = ["router"]
