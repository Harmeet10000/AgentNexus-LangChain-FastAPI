# ── from credit_admin_router.py ──
"""Admin/Portal credit endpoints."""

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query, Response, status
from returns.result import Failure, Success

from app.features.auth.api.http.dependencies import (
    AuthorizationContext,
    CurrentVerifiedUser,
    require_role,
)
from app.features.auth.domain.user import UserRole
from app.kernel.result import render_result
from app.utils import APIResponse

from .dependencies import ApplyCreditsToInvoiceDep, CreditServiceDep
from .dto import (
    ConsumedCredit,
    CreditBalanceResponse,
    CreditConsumptionResult,
    CreditGrantDTO,
    CreditGrantResponse,
    CreditHistoryResponse,
)
from .exceptions import CreditAuthorizationError, CreditMetadataError

if TYPE_CHECKING:
    from .apply import CreditApplication
    from .exceptions import CreditResult

admin_router = APIRouter(prefix="/credits", tags=["credits"])


@admin_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
async def grant_credit(
    dto: CreditGrantDTO,
    service: CreditServiceDep,
    authorization: Annotated[
        AuthorizationContext,
        Depends(require_role(UserRole.ADMIN)),
    ],
    response: Response,
) -> APIResponse[CreditGrantResponse]:
    """Grant credit to a user (Requirement 49). Admin only."""
    # ``authorization`` is intentionally resolved even though the target is
    # supplied in the payload: it is the live-account authority gate for this
    # privileged mutation, not a JWT role/permission check.
    _ = authorization
    target_user_id = dto.metadata_.get("target_user_id")
    if not target_user_id:
        return render_result(
            Failure(CreditMetadataError(message="target_user_id is required")),
            response,
            message="Credit grant failed",
        )
    result = await service.grant_credit(user_id=target_user_id, dto=dto)
    return render_result(
        result, response, message="Credit granted", success_status=status.HTTP_201_CREATED
    )


@admin_router.get("/balance/{user_id}")
async def get_credit_balance(
    user_id: Annotated[str, Path(min_length=1)],
    service: CreditServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
) -> APIResponse[CreditBalanceResponse]:
    """Get user's total available credit balance (Requirement 52.1)."""
    msg = "Cannot view other users' credit balances"
    if str(user.id) != user_id and user.role != UserRole.ADMIN:
        return render_result(
            Failure(CreditAuthorizationError(message=msg)), response, message="Credit access denied"
        )
    result = await service.get_credit_balance(user_id)
    return render_result(result, response, message="Credit balance")


@admin_router.get("/history/{user_id}")
async def get_credit_history(
    user_id: Annotated[str, Path(min_length=1)],
    service: CreditServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    *,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[CreditHistoryResponse]:
    """Get user's credit and consumption history (Requirement 52.2)."""
    msg = "Cannot view other users' credit history"
    if str(user.id) != user_id and user.role != UserRole.ADMIN:
        return render_result(
            Failure(CreditAuthorizationError(message=msg)), response, message="Credit access denied"
        )
    result = await service.get_credit_history(user_id, limit=limit, offset=offset)
    return render_result(result, response, message="Credit history")


# ── from credit_internal_router.py ──
"""System-internal credit endpoints (called by InvoiceService)."""


internal_router = APIRouter(prefix="/credits", tags=["credits-internal"])


@internal_router.post("/apply-to-invoice")
async def apply_credit_to_invoice(
    apply_op: ApplyCreditsToInvoiceDep,
    user: CurrentVerifiedUser,
    invoice_id: Annotated[str, Query(min_length=1)],
    response: Response,
) -> APIResponse[CreditConsumptionResult]:
    """Apply available credits to an invoice (Requirement 50, 55).

    Deep operation that owns the complete transaction:
    - Loads and locks the invoice authoritatively
    - Verifies ownership and status
    - Obtains the authoritative invoice amount (no caller-supplied amount)
    - Selects and applies credits deterministically
    - Records consumption and updates balances
    - Updates invoice status if fully covered
    - Commits or rolls back atomically
    """
    raw_result = await apply_op.execute(invoice_id=UUID(invoice_id), user_id=str(user.id))
    if isinstance(raw_result, Failure):
        # Construct a properly-typed failure result for render_result
        failure_result: CreditResult[CreditConsumptionResult] = Failure(raw_result.failure())
        return render_result(failure_result, response, message="Credit application failed")
    application: CreditApplication = raw_result.unwrap()
    # Convert to existing DTO for API compatibility
    dto_consumed = [
        ConsumedCredit(
            credit_id=c.credit_id,
            consumed_amount=c.consumed_amount,
            remaining_balance=c.remaining_balance,
        )
        for c in application.credits_consumed
    ]
    return render_result(
        Success(
            CreditConsumptionResult(
                credit_applied=application.credit_applied_paisa,
                credit_applied_rupees=application.credit_applied_rupees,
                cash_due=application.cash_due_paisa,
                cash_due_rupees=application.cash_due_rupees,
                credits_consumed=dto_consumed,
                invoice_paid_in_full=application.invoice_paid_in_full,
            )
        ),
        response,
        message="Credit applied",
    )
