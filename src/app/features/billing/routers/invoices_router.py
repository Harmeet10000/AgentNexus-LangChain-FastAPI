"""Invoice endpoints (user-scoped)."""

from typing import Annotated, cast

from fastapi import APIRouter, Path, Query

from app.features.auth import CurrentVerifiedUser
from app.features.billing.dto import InvoiceResponse, VoidInvoiceDTO
from app.utils import APIResponse, http_response

from ..dependencies import InvoiceServiceDep

router = APIRouter(prefix="/invoices", tags=["billing-invoices"])


@router.get("")
async def list_invoices(  # noqa: PLR0917
    service: InvoiceServiceDep,
    user: CurrentVerifiedUser,
    subscription_id: Annotated[str | None, Query(alias="subscriptionId")] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> APIResponse[list[InvoiceResponse]]:
    result = await service.list_invoices(
        str(user.id),
        subscription_id=subscription_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    if isinstance(result, APIResponse):
        return cast("APIResponse[list[InvoiceResponse]]", result)
    return http_response(message="Invoices", data=result)


@router.get("/{invoice_id}")
async def get_invoice(
    invoice_id: Annotated[str, Path(min_length=1)],
    service: InvoiceServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[InvoiceResponse]:
    result = await service.get_invoice(invoice_id, user_id=str(user.id))
    if isinstance(result, APIResponse):
        return cast("APIResponse[InvoiceResponse]", result)
    return http_response(message="Invoice", data=result)


@router.post("/{invoice_id}/void")
async def void_invoice(
    invoice_id: Annotated[str, Path(min_length=1)],
    payload: VoidInvoiceDTO,
    service: InvoiceServiceDep,
    user: CurrentVerifiedUser,
) -> APIResponse[InvoiceResponse]:
    voided = await service.void_invoice(invoice_id, payload, user_id=str(user.id))
    if isinstance(voided, APIResponse):
        return cast("APIResponse[InvoiceResponse]", voided)
    return http_response(message="Invoice voided", data=voided)
