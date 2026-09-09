"""System-internal credit endpoints (called by InvoiceService)."""

from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from returns.result import Failure

from app.connections import get_postgres_db
from app.features.auth import CurrentVerifiedUser
from app.features.billing.invoices.dependencies import InvoiceServiceDep
from app.shared.result import render_result
from app.utils import APIResponse

from ..dependencies import CreditServiceDep
from ..dto.consumption_dto import CreditConsumptionResult
from ..errors import CreditCollaboratorError

router = APIRouter(prefix="/credits", tags=["credits-internal"])


@router.post("/apply-to-invoice")
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
