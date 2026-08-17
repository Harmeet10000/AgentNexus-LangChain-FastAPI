"""Payment endpoints (user-scoped)."""

from typing import Annotated, cast

from fastapi import APIRouter, Path, Query

from app.features.auth import CurrentClaims, CurrentVerifiedUser
from app.features.billing.dto import PaymentResponse, RefundRequestDTO, RefundResponse
from app.utils import APIResponse, http_response

from ..dependencies import PaymentServiceDep, SubscriptionServiceDep

router = APIRouter(tags=["billing-payments"])


@router.get("/subscriptions/{subscription_id}/payments")
async def list_payments(  # noqa: PLR0917
    subscription_id: Annotated[str, Path(min_length=1)],
    service: PaymentServiceDep,
    sub_service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[list[PaymentResponse]]:
    sub = await sub_service.get_subscription(str(user.id), subscription_id)
    if isinstance(sub, APIResponse):
        return cast("APIResponse[list[PaymentResponse]]", sub)
    result = await service.list_payments(sub.id, limit=limit, offset=offset)
    if isinstance(result, APIResponse):
        return cast("APIResponse[list[PaymentResponse]]", result)
    return http_response(message="Payments", data=result)


@router.get("/payments/{payment_id}")
async def get_payment(
    payment_id: Annotated[str, Path(min_length=1)],
    service: PaymentServiceDep,
) -> APIResponse[PaymentResponse]:
    result = await service.get_payment(payment_id)
    if isinstance(result, APIResponse):
        return cast("APIResponse[PaymentResponse]", result)
    return http_response(message="Payment", data=result)


@router.post("/payments/{payment_id}/refund")
async def refund_payment(
    payment_id: Annotated[str, Path(min_length=1)],
    payload: RefundRequestDTO,
    service: PaymentServiceDep,
    claims: CurrentClaims,
) -> APIResponse[RefundResponse]:
    refund = await service.refund(payment_id, payload, user_id=claims.sub)
    if isinstance(refund, APIResponse):
        return cast("APIResponse[RefundResponse]", refund)
    return http_response(message="Refund issued", data=refund)
