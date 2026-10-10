# ── from credit_admin_router.py ──
"""Admin/Portal credit endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response, status
from returns.result import Failure

from app.features.auth.dependencies import CurrentClaims, require_role
from app.features.auth.models import UserRole
from app.shared.result import render_result
from app.utils import APIResponse

from .dependencies import CreditServiceDep
from .dto import (
    CreditBalanceResponse,
    CreditGrantDTO,
    CreditGrantResponse,
    CreditHistoryResponse,
)
from .exceptions import CreditAuthorizationError, CreditMetadataError

admin_router = APIRouter(prefix="/credits", tags=["credits"])


@admin_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
async def grant_credit(
    dto: CreditGrantDTO,
    service: CreditServiceDep,
    claims: CurrentClaims,  # noqa: ARG001 — FastAPI DI dependency
    response: Response,
) -> APIResponse[CreditGrantResponse]:
    """Grant credit to a user (Requirement 49). Admin only."""
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
    claims: CurrentClaims,
    response: Response,
) -> APIResponse[CreditBalanceResponse]:
    """Get user's total available credit balance (Requirement 52.1)."""
    msg = "Cannot view other users' credit balances"
    if claims.sub != user_id and claims.role != UserRole.ADMIN.value:
        return render_result(
            Failure(CreditAuthorizationError(message=msg)), response, message="Credit access denied"
        )
    result = await service.get_credit_balance(user_id)
    return render_result(result, response, message="Credit balance")


@admin_router.get("/history/{user_id}")
async def get_credit_history(
    user_id: Annotated[str, Path(min_length=1)],
    service: CreditServiceDep,
    claims: CurrentClaims,
    response: Response,
    *,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[CreditHistoryResponse]:
    """Get user's credit and consumption history (Requirement 52.2)."""
    msg = "Cannot view other users' credit history"
    if claims.sub != user_id and claims.role != UserRole.ADMIN.value:
        return render_result(
            Failure(CreditAuthorizationError(message=msg)), response, message="Credit access denied"
        )
    result = await service.get_credit_history(user_id, limit=limit, offset=offset)
    return render_result(result, response, message="Credit history")

# ── from credit_internal_router.py ──
"""System-internal credit endpoints (called by InvoiceService)."""

from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from returns.result import Failure

from app.connections import get_postgres_db
from app.features.auth.dependencies import CurrentVerifiedUser
from app.features.billing.invoices.dependencies import InvoiceServiceDep
from app.shared.result import render_result
from app.utils import APIResponse

from .dependencies import CreditServiceDep
from .dto import CreditConsumptionResult
from .exceptions import CreditCollaboratorError

internal_router = APIRouter(prefix="/credits", tags=["credits-internal"])


@internal_router.post("/apply-to-invoice")
async def apply_credit_to_invoice(  # noqa: PLR0917 - endpoint needs services, identity, params, response
    service: CreditServiceDep,
    invoice_service: InvoiceServiceDep,
    user: CurrentVerifiedUser,
    invoice_id: Annotated[str, Query(min_length=1)],
    invoice_gross_total: Annotated[Decimal, Query(gt=0)],
    response: Response,
    *,
    session=Depends(get_postgres_db),
) -> APIResponse[CreditConsumptionResult]:
    """Apply available credits to an invoice (Requirement 50, 55).

    System/internal endpoint called by InvoiceService during invoice generation.
    The session is owned by InvoiceService — this endpoint does NOT commit.

    The caller is authenticated and the invoice must belong to them: a
    caller-supplied user id is never trusted, so credits cannot be drained
    across accounts.
    """
    invoice_result = await invoice_service.get_invoice(invoice_id, user_id=str(user.id))
    if isinstance(invoice_result, Failure):
        error = invoice_result.failure()
        return render_result(
            Failure(CreditCollaboratorError(message=error.message, details=error.details)),
            response,
            message="Credit applied",
        )
    result = await service.consume_credits(
        user_id=str(user.id),
        invoice_id=UUID(invoice_id),
        invoice_gross_total=invoice_gross_total,
        session=session,
    )
    return render_result(result, response, message="Credit applied")
