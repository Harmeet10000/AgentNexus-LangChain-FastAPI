"""Invoice endpoints (user-scoped)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response

from app.features.auth.api.http.dependencies import CurrentVerifiedUser
from app.kernel.result import render_result
from app.shared.rate_limit import get_rate_limiter
from app.utils import APIResponse

from .dependencies import InvoiceServiceDep
from .dto import InvoiceListResponse, InvoiceResponse, VoidInvoiceDTO

# Invoice voids are rare corrective money movements; tight per-IP guard.
limit_invoice_void = get_rate_limiter(burst=5, rate=5, period=300, scope="invoice-void")

router = APIRouter(prefix="/invoices", tags=["billing-invoices"])


@router.get("")
async def list_invoices(  # noqa: PLR0917
    service: InvoiceServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
    subscription_id: Annotated[str | None, Query(alias="subscriptionId")] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[InvoiceListResponse]:
    result = await service.list_invoices(
        str(user.id),
        subscription_id=subscription_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    return render_result(result, response, message="Invoices")


@router.get("/{invoice_id}")
async def get_invoice(
    invoice_id: Annotated[str, Path(min_length=1)],
    service: InvoiceServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
) -> APIResponse[InvoiceResponse]:
    result = await service.get_invoice(invoice_id, user_id=str(user.id))
    return render_result(result, response, message="Invoice")


@router.post("/{invoice_id}/void", dependencies=[Depends(limit_invoice_void)])
async def void_invoice(
    invoice_id: Annotated[str, Path(min_length=1)],
    payload: VoidInvoiceDTO,
    service: InvoiceServiceDep,
    user: CurrentVerifiedUser,
    response: Response,
) -> APIResponse[InvoiceResponse]:
    result = await service.void_invoice(invoice_id, payload, user_id=str(user.id))
    return render_result(result, response, message="Invoice voided")
