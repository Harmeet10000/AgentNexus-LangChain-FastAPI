"""Subscription lifecycle endpoints (user-scoped)."""

from typing import Annotated

from fastapi import APIRouter, Header, Path, Query, Request, Response, status

from app.features.auth.dependencies import CurrentVerifiedUser
from app.features.billing.subscriptions.models import SubscriptionStatus
from app.kernel.result import render_result
from app.utils import APIResponse

from .dependencies import SubscriptionServiceDep
from .dto import (
    PlanChangeDTO,
    ProrationCalculation,
    SubscriptionCancelDTO,
    SubscriptionCreateDTO,
    SubscriptionListResponse,
    SubscriptionPauseDTO,
    SubscriptionResponse,
)

router = APIRouter(prefix="/subscriptions", tags=["billing-subscriptions"])


def _request_idempotency_key(request: Request, header: str | None) -> str | None:
    """Prefer the explicit retry key, then the existing request correlation."""
    return header or getattr(request.state, "correlation_id", None)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_subscription(  # noqa: PLR0917
    payload: SubscriptionCreateDTO,
    request: Request,
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> APIResponse[SubscriptionResponse]:
    result = await service.create_subscription(
        str(user.id), payload, idempotency_key=_request_idempotency_key(request, idempotency_key)
    )
    return render_result(
        result, response, message="Subscription created", success_status=status.HTTP_201_CREATED
    )


@router.get("")
async def list_subscriptions(  # noqa: PLR0917
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    status_filter: Annotated[SubscriptionStatus | None, Query(alias="status")] = None,
    plan_id: Annotated[str | None, Query(alias="planId")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[SubscriptionListResponse]:
    result = await service.list_subscriptions(
        str(user.id), status=status_filter, plan_id=plan_id, limit=limit, offset=offset
    )
    return render_result(
        result, response, message="Subscriptions", success_status=status.HTTP_200_OK
    )


@router.get("/{subscription_id}")
async def get_subscription(
    subscription_id: Annotated[str, Path(min_length=1)],
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
) -> APIResponse[SubscriptionResponse]:
    result = await service.get_subscription(str(user.id), subscription_id)
    return render_result(
        result, response, message="Subscription", success_status=status.HTTP_200_OK
    )


@router.post("/{subscription_id}/cancel")
async def cancel_subscription(  # noqa: PLR0917
    subscription_id: Annotated[str, Path(min_length=1)],
    payload: SubscriptionCancelDTO,
    request: Request,
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> APIResponse[SubscriptionResponse]:
    result = await service.cancel_subscription(
        str(user.id),
        subscription_id,
        payload,
        idempotency_key=_request_idempotency_key(request, idempotency_key),
    )
    return render_result(
        result, response, message="Subscription cancelled", success_status=status.HTTP_200_OK
    )


@router.post("/{subscription_id}/pause")
async def pause_subscription(  # noqa: PLR0917
    subscription_id: Annotated[str, Path(min_length=1)],
    payload: SubscriptionPauseDTO,
    request: Request,
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> APIResponse[SubscriptionResponse]:
    result = await service.pause_subscription(
        str(user.id),
        subscription_id,
        payload,
        idempotency_key=_request_idempotency_key(request, idempotency_key),
    )
    return render_result(
        result, response, message="Subscription paused", success_status=status.HTTP_200_OK
    )


@router.post("/{subscription_id}/resume")
async def resume_subscription(  # noqa: PLR0917
    subscription_id: Annotated[str, Path(min_length=1)],
    service: SubscriptionServiceDep,
    request: Request,
    user: CurrentVerifiedUser,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> APIResponse[SubscriptionResponse]:
    result = await service.resume_subscription(
        str(user.id),
        subscription_id,
        idempotency_key=_request_idempotency_key(request, idempotency_key),
    )
    return render_result(
        result, response, message="Subscription resumed", success_status=status.HTTP_200_OK
    )


@router.post("/{subscription_id}/change-plan")
async def change_plan(  # noqa: PLR0917
    subscription_id: Annotated[str, Path(min_length=1)],
    payload: PlanChangeDTO,
    request: Request,
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> APIResponse[SubscriptionResponse]:
    result = await service.change_plan(
        str(user.id),
        subscription_id,
        payload,
        idempotency_key=_request_idempotency_key(request, idempotency_key),
    )
    return render_result(
        result, response, message="Plan changed", success_status=status.HTTP_200_OK
    )


@router.get("/{subscription_id}/change-preview")
async def change_preview(
    subscription_id: Annotated[str, Path(min_length=1)],
    new_plan_id: Annotated[str, Query(alias="newPlanId")],
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
) -> APIResponse[ProrationCalculation]:
    result = await service.get_change_preview(str(user.id), subscription_id, new_plan_id)
    return render_result(
        result, response, message="Proration preview", success_status=status.HTTP_200_OK
    )


@router.post("/{subscription_id}/trial-extension")
async def request_trial_extension(  # noqa: PLR0917
    subscription_id: Annotated[str, Path(min_length=1)],
    service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    days: Annotated[int, Query(ge=1, le=30)] = 7,
    reason: Annotated[str | None, Query(max_length=500)] = None,
) -> APIResponse[dict[str, object]]:
    result = await service.request_trial_extension(
        str(user.id), subscription_id, days=days, reason=reason
    )
    return render_result(
        result, response, message="Trial extension requested", success_status=status.HTTP_200_OK
    )
