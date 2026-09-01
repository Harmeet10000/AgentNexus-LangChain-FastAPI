"""System-internal credit endpoints (called by InvoiceService)."""

from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response

from app.connections import get_postgres_db
from app.shared.result import render_result
from app.utils import APIResponse

from ..dependencies import CreditServiceDep
from ..dto.consumption_dto import CreditConsumptionResult

router = APIRouter(prefix="/credits", tags=["credits-internal"])


@router.post("/apply-to-invoice")
async def apply_credit_to_invoice(
    service: CreditServiceDep,
    user_id: Annotated[str, Query(min_length=1)],
    invoice_id: Annotated[str, Query(min_length=1)],
    invoice_gross_total: Annotated[Decimal, Query(gt=0)],
    response: Response,
    *,
    session=Depends(get_postgres_db),
) -> APIResponse[CreditConsumptionResult]:
    """Apply available credits to an invoice (Requirement 50, 55).

    System/internal endpoint called by InvoiceService during invoice generation.
    The session is owned by InvoiceService — this endpoint does NOT commit.
    """
    result = await service.consume_credits(
        user_id=user_id,
        invoice_id=UUID(invoice_id),
        invoice_gross_total=invoice_gross_total,
        session=session,
    )
    return render_result(result, response, message="Credit applied")
