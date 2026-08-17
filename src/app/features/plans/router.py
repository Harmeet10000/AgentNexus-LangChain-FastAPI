"""Plan endpoints: public listing + admin management."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, status

from app.features.auth import CurrentClaims, require_role
from app.features.auth.model import UserRole
from app.utils import APIResponse, http_response

from .dependencies import PlanServiceDep
from .dto import PlanCreateDTO, PlanResponse, PlanUpdateDTO

router = APIRouter(prefix="/plans", tags=["billing-plans"])


@router.get("")
async def list_plans(
    service: PlanServiceDep,
    include_inactive: Annotated[bool, Query(alias="includeInactive")] = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[list[PlanResponse]]:
    result = await service.list_plans(include_inactive=include_inactive, limit=limit, offset=offset)
    return http_response(message="Plans", data=result)


@router.get("/{plan_id}")
async def get_plan(
    plan_id: Annotated[str, Path(min_length=1)],
    service: PlanServiceDep,
) -> APIResponse[PlanResponse]:
    result = await service.get_plan(plan_id)
    return http_response(message="Plan", data=result)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def create_plan(
    payload: PlanCreateDTO,
    service: PlanServiceDep,
    claims: CurrentClaims,
) -> APIResponse[PlanResponse]:
    result = await service.create_plan(payload, user_id=claims.sub)
    return http_response(message="Plan created", data=result, status_code=status.HTTP_201_CREATED)


@router.patch(
    "/{plan_id}",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def update_plan(
    plan_id: Annotated[str, Path(min_length=1)],
    payload: PlanUpdateDTO,
    service: PlanServiceDep,
    claims: CurrentClaims,
) -> APIResponse[PlanResponse]:
    result = await service.update_plan(plan_id, payload, user_id=claims.sub)
    return http_response(message="Plan updated", data=result)


@router.post(
    "/{plan_id}/archive",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def archive_plan(
    plan_id: Annotated[str, Path(min_length=1)],
    service: PlanServiceDep,
    claims: CurrentClaims,
) -> APIResponse[PlanResponse]:
    result = await service.archive_plan(plan_id, user_id=claims.sub)
    return http_response(message="Plan archived", data=result)
