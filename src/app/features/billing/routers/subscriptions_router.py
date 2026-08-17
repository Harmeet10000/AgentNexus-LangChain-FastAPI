"""Subscription lifecycle endpoints (user-scoped)."""

from typing import Annotated, cast

from fastapi import APIRouter, Path, Query, status

from app.features.auth import CurrentVerifiedUser
from app.features.billing.dto import (
    PlanChangeDTO,
    ProrationCalculation,
    SubscriptionCancelDTO,
    SubscriptionCreateDTO,
    SubscriptionListResponse,
    SubscriptionPauseDTO,
    SubscriptionResponse,
)
from app.features.billing.models import SubscriptionStatus
from app.utils import APIResponse, http_response

from ..dependencies import SubscriptionServiceDep

router = APIRouter(prefix="/subscriptions", tags=["billing-subscriptions"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_subscription(
    payload: SubscriptionCreateDTO,
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[SubscriptionResponse]:
    result = await service.create_subscription(str(user.id), payload)
    if isinstance(result, APIResponse):
        return cast("APIResponse[SubscriptionResponse]", result)
    return http_response(
        message="Subscription created", data=result, status_code=status.HTTP_201_CREATED
    )


@router.get("")
async def list_subscriptions(  # noqa: PLR0917
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    status_filter: Annotated[SubscriptionStatus | None, Query(alias="status")] = None,
    plan_id: Annotated[str | None, Query(alias="planId")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[SubscriptionListResponse]:
    result = await service.list_subscriptions(
        str(user.id), status=status_filter, plan_id=plan_id, limit=limit, offset=offset
    )
    if isinstance(result, APIResponse):
        return cast("APIResponse[SubscriptionListResponse]", result)
    return http_response(message="Subscriptions", data=result)


@router.get("/{subscription_id}")
async def get_subscription(
    subscription_id: Annotated[str, Path(min_length=1)],
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[SubscriptionResponse]:
    result = await service.get_subscription(str(user.id), subscription_id)
    if isinstance(result, APIResponse):
        return cast("APIResponse[SubscriptionResponse]", result)
    return http_response(message="Subscription", data=result)


@router.post("/{subscription_id}/cancel")
async def cancel_subscription(
    subscription_id: Annotated[str, Path(min_length=1)],
    payload: SubscriptionCancelDTO,
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[SubscriptionResponse]:
    result = await service.cancel_subscription(str(user.id), subscription_id, payload)
    if isinstance(result, APIResponse):
        return cast("APIResponse[SubscriptionResponse]", result)
    return http_response(message="Subscription cancelled", data=result)


@router.post("/{subscription_id}/pause")
async def pause_subscription(
    subscription_id: Annotated[str, Path(min_length=1)],
    payload: SubscriptionPauseDTO,
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[SubscriptionResponse]:
    result = await service.pause_subscription(str(user.id), subscription_id, payload)
    if isinstance(result, APIResponse):
        return cast("APIResponse[SubscriptionResponse]", result)
    return http_response(message="Subscription paused", data=result)


@router.post("/{subscription_id}/resume")
async def resume_subscription(
    subscription_id: Annotated[str, Path(min_length=1)],
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[SubscriptionResponse]:
    result = await service.resume_subscription(str(user.id), subscription_id)
    if isinstance(result, APIResponse):
        return cast("APIResponse[SubscriptionResponse]", result)
    return http_response(message="Subscription resumed", data=result)


@router.post("/{subscription_id}/change-plan")
async def change_plan(
    subscription_id: Annotated[str, Path(min_length=1)],
    payload: PlanChangeDTO,
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[SubscriptionResponse]:
    result = await service.change_plan(str(user.id), subscription_id, payload)
    if isinstance(result, APIResponse):
        return cast("APIResponse[SubscriptionResponse]", result)
    return http_response(message="Plan changed", data=result)


@router.get("/{subscription_id}/change-preview")
async def change_preview(
    subscription_id: Annotated[str, Path(min_length=1)],
    new_plan_id: Annotated[str, Query(alias="newPlanId")],
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[ProrationCalculation]:
    result = await service.get_change_preview(str(user.id), subscription_id, new_plan_id)
    if isinstance(result, APIResponse):
        return cast("APIResponse[ProrationCalculation]", result)
    return http_response(message="Proration preview", data=result)


@router.post("/{subscription_id}/trial-extension")
async def request_trial_extension(
    subscription_id: Annotated[str, Path(min_length=1)],
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    days: Annotated[int, Query(ge=1, le=30)] = 7,
    reason: Annotated[str | None, Query(max_length=500)] = None,
) -> APIResponse[dict[str, object]]:
    result = await service.request_trial_extension(
        str(user.id), subscription_id, days=days, reason=reason
    )
    if isinstance(result, APIResponse):
        return cast("APIResponse[dict[str, object]]", result)
    return http_response(message="Trial extension requested", data=result)
