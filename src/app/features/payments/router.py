"""Payment endpoints (user-scoped)."""

from typing import Annotated

from fastapi import APIRouter, Path, Query, Response, status
from returns.result import Failure

from app.features.auth import CurrentClaims, CurrentVerifiedUser
from app.features.subscriptions.dependencies import SubscriptionServiceDep
from app.shared.result import render_result
from app.utils import APIResponse

from .dependencies import PaymentServiceDep
from .dto import PaymentResponse, RefundRequestDTO, RefundResponse
from .errors import PaymentCollaboratorError

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
    response: Response,
) -> APIResponse[PaymentResponse]:
    result = await service.get_payment(payment_id)
    return render_result(result, response, message="Payment")


@router.post("/payments/{payment_id}/refund")
async def refund_payment(
    payment_id: Annotated[str, Path(min_length=1)],
    payload: RefundRequestDTO,
    service: PaymentServiceDep,
    claims: CurrentClaims,
    response: Response,
) -> APIResponse[RefundResponse]:
    refund = await service.refund(payment_id, payload, user_id=claims.sub)
    return render_result(
        refund, response, message="Refund issued", success_status=status.HTTP_200_OK
    )
