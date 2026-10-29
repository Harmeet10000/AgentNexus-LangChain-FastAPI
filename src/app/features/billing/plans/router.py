"""Plan endpoints: public listing + admin management."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Query, Response, status

from app.features.auth.api.http.dependencies import AuthorizationContext, require_role
from app.features.auth.domain.user import UserRole
from app.kernel.result import render_result
from app.utils import APIResponse

from .dependencies import PlanServiceDep
from .dto import PlanCreateDTO, PlanResponse, PlanUpdateDTO

router = APIRouter(prefix="/plans", tags=["billing-plans"])


@router.get("")
async def list_plans(
    service: PlanServiceDep,
    response: Response,
    include_inactive: Annotated[bool, Query(alias="includeInactive")] = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[list[PlanResponse]]:
    result = await service.list_plans(include_inactive=include_inactive, limit=limit, offset=offset)
    return render_result(result, response, message="Plans")


@router.get("/{plan_id}")
async def get_plan(
    plan_id: Annotated[str, Path(min_length=1)],
    service: PlanServiceDep,
    response: Response,
) -> APIResponse[PlanResponse]:
    result = await service.get_plan(plan_id)
    return render_result(result, response, message="Plan")


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
async def create_plan(
    payload: PlanCreateDTO,
    service: PlanServiceDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_role(UserRole.ADMIN)),
    ],
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> APIResponse[PlanResponse]:
    result = await service.create_plan(
        payload, user_id=authorization.user_id, idempotency_key=idempotency_key
    )
    return render_result(
        result, response, message="Plan created", success_status=status.HTTP_201_CREATED
    )


@router.patch(
    "/{plan_id}",
)
async def update_plan(  # noqa: PLR0917 — FastAPI's explicit endpoint dependency parameters
    plan_id: Annotated[str, Path(min_length=1)],
    payload: PlanUpdateDTO,
    service: PlanServiceDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_role(UserRole.ADMIN)),
    ],
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> APIResponse[PlanResponse]:
    result = await service.update_plan(
        plan_id, payload, user_id=authorization.user_id, idempotency_key=idempotency_key
    )
    return render_result(result, response, message="Plan updated")


@router.post(
    "/{plan_id}/archive",
)
async def archive_plan(
    plan_id: Annotated[str, Path(min_length=1)],
    service: PlanServiceDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_role(UserRole.ADMIN)),
    ],
    response: Response,
) -> APIResponse[PlanResponse]:
    result = await service.archive_plan(plan_id, user_id=authorization.user_id)
    return render_result(result, response, message="Plan archived")
