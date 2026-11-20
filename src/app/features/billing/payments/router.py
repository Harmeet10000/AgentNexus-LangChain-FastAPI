"""Payment endpoints (user-scoped)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Query, Response, status
from returns.result import Failure

from app.features.auth.api.http.dependencies import CurrentVerifiedUser
from app.features.billing.subscriptions.dependencies import SubscriptionServiceDep
from app.shared.rate_limit import get_rate_limiter
from app.shared.result.render import APIResponse, render_result

from .dependencies import PaymentServiceDep
from .dto import PaymentResponse, RefundRequestDTO, RefundResponse
from .exceptions import PaymentCollaboratorError

# Refunds are rare, irreversible-ish money movements; tight per-IP guard.
limit_refund = get_rate_limiter(burst=5, rate=5, period=300, scope="payment-refund")

router = APIRouter(tags=["billing-payments"])


@router.get("/subscriptions/{subscription_id}/payments")
async def list_payments(  # noqa: PLR0917
    subscription_id: Annotated[str, Path(min_length=1)],
    service: PaymentServiceDep,
    sub_service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[list[PaymentResponse]]:
    sub_result = await sub_service.get_subscription(str(user.id), subscription_id)
    if isinstance(sub_result, Failure):
        error = sub_result.failure()
        return render_result(
            Failure(PaymentCollaboratorError(message=error.message, details=error.details)),
            response,
            message="Payments",
        )
    result = await service.list_payments(sub_result.unwrap().id, limit=limit, offset=offset)
    return render_result(result, response, message="Payments")


@router.get("/payments/{payment_id}")
async def get_payment(
    payment_id: Annotated[str, Path(min_length=1)],
    service: PaymentServiceDep,
    sub_service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
) -> APIResponse[PaymentResponse]:
    result = await service.get_payment(payment_id)
    if isinstance(result, Failure):
        return render_result(result, response, message="Payment")
    payment = result.unwrap()
    ownership = await sub_service.get_subscription(str(user.id), payment.subscription_id)
    if isinstance(ownership, Failure):
        error = ownership.failure()
        return render_result(
            Failure(PaymentCollaboratorError(message=error.message, details=error.details)),
            response,
            message="Payment",
        )
    return render_result(result, response, message="Payment")


@router.post("/payments/{payment_id}/refund", dependencies=[Depends(limit_refund)])
async def refund_payment(  # noqa: PLR0917 - endpoint needs path payload, services, claims, response
    payment_id: Annotated[str, Path(min_length=1)],
    payload: RefundRequestDTO,
    service: PaymentServiceDep,
    sub_service: SubscriptionServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> APIResponse[RefundResponse]:
    resolved = await service.get_payment(payment_id)
    if isinstance(resolved, Failure):
        return render_result(
            resolved, response, message="Refund issued", success_status=status.HTTP_200_OK
        )
    user_id = str(user.id)
    ownership = await sub_service.get_subscription(user_id, resolved.unwrap().subscription_id)
    if isinstance(ownership, Failure):
        error = ownership.failure()
        return render_result(
            Failure(PaymentCollaboratorError(message=error.message, details=error.details)),
            response,
            message="Refund issued",
            success_status=status.HTTP_200_OK,
        )
    refund = await service.refund(
        payment_id,
        payload,
        user_id=user_id,
        idempotency_key=idempotency_key,
    )
    return render_result(
        refund, response, message="Refund issued", success_status=status.HTTP_200_OK
    )
