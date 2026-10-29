"""Tenant membership administration endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response
from returns.result import Failure, Success

from app.kernel.result import render_result
from app.utils import APIResponse

from .dependencies import (
    TenantMembershipServiceDep,
    require_tenant_membership_admin,
)
from .dto import UpdateSelfReviewRequest, UpdateTenantMembershipRequest

router = APIRouter(prefix="/tenants", tags=["tenants"])
_require_tenant_admin = Depends(require_tenant_membership_admin())


@router.patch(
    "/{tenant_id}/members/{user_id}/legal-roles",
    dependencies=[_require_tenant_admin],
)
async def update_tenant_member_roles(
    tenant_id: Annotated[str, Path(min_length=1, max_length=255)],
    user_id: Annotated[str, Path(min_length=1, max_length=255)],
    body: UpdateTenantMembershipRequest,
    service: TenantMembershipServiceDep,
    response: Response,
) -> APIResponse[object]:
    result = await service.update_roles(
        tenant_id=tenant_id,
        user_id=user_id,
        legal_roles=body.legal_roles,
        expected_revision=body.expected_revision,
    )
    if isinstance(result, Failure):
        return render_result(result, response, message="Tenant membership update failed")
    payload = {
        "tenantId": tenant_id,
        "userId": user_id,
        "legalRoles": [role.value for role in result.unwrap()],
    }
    return render_result(
        Success(payload),
        response,
        message="Tenant membership updated",
    )


@router.patch(
    "/{tenant_id}/settings/self-review",
    dependencies=[_require_tenant_admin],
)
async def update_tenant_self_review(
    tenant_id: Annotated[str, Path(min_length=1, max_length=255)],
    body: UpdateSelfReviewRequest,
    service: TenantMembershipServiceDep,
    response: Response,
) -> APIResponse[object]:
    result = await service.set_self_review_allowed(tenant_id=tenant_id, allowed=body.allowed)
    if isinstance(result, Failure):
        return render_result(result, response, message="Tenant setting update failed")
    return render_result(
        Success({"tenantId": tenant_id, "selfReviewAllowed": result.unwrap()}),
        response,
        message="Tenant self-review setting updated",
    )
