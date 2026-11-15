"""F11 regression: GET /invoices must render the same envelope as /subscriptions.

The smoke found ``GET /api/v2/invoices`` returning a bare array while sibling
``GET /api/v2/subscriptions`` returned the paginated
``{items, total, limit, offset}`` envelope. The invoices list now flows through
an ``InvoiceListResponse`` built on the repository's ``(items, total_count)``
pagination contract, mirroring the subscriptions list.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from returns.result import Success

from app.features.billing.invoices.dto import InvoiceListResponse
from app.features.billing.invoices.models import Invoice
from app.features.billing.invoices.router import router as invoices_router
from app.features.billing.invoices.service import InvoiceService
from app.features.billing.subscriptions.dto import SubscriptionListResponse
from app.kernel.result import APIResponse

if TYPE_CHECKING:
    from app.features.billing.invoices.models import Invoice as InvoiceModel

pytestmark = pytest.mark.unit


class _FakeInvoiceRepository:
    """Duck-typed InvoiceRepository returning one committed invoice row."""

    def __init__(self, invoice: InvoiceModel, total: int) -> None:
        self._invoice = invoice
        self._total = total

    async def list_by_user(
        self,
        user_id: str,
        *,
        subscription_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Success[tuple[list[InvoiceModel], int]]:
        del user_id, subscription_id, status
        return Success(([self._invoice] if offset == 0 else [], self._total))


def _invoice() -> InvoiceModel:
    return Invoice(
        id=uuid4(),
        invoice_number="INV-2026-0001",
        subscription_id=uuid4(),
        user_id="user-1",
        status="issued",
        subtotal=Decimal("100.00"),
        tax_rate=Decimal("0.18"),
        tax_amount=Decimal("18.00"),
        total=Decimal("118.00"),
        credit_applied=Decimal(0),
        cgst_amount=Decimal(0),
        sgst_amount=Decimal(0),
        igst_amount=Decimal("18.00"),
        sac_code="998314",
        currency="INR",
        seller_gstin="06ABCDE1234F1Z5",
        place_of_supply="06",
        issued_at=datetime.now(tz=UTC),
        created_at=datetime.now(tz=UTC),
    )


def test_invoice_list_envelope_matches_subscription_envelope() -> None:
    invoice_list = InvoiceListResponse(items=[], total=0, limit=50, offset=0)
    subscription_list = SubscriptionListResponse(items=[], total=0, limit=50, offset=0)

    assert set(invoice_list.model_dump()) == set(subscription_list.model_dump())


def test_list_invoices_service_returns_paginated_envelope() -> None:
    invoice = _invoice()
    service = InvoiceService(
        session=None,  # type: ignore[arg-type] — list path never touches the session
        invoices=_FakeInvoiceRepository(invoice, total=7),  # type: ignore[arg-type]
        subscriptions=None,  # type: ignore[arg-type]
        plans=None,  # type: ignore[arg-type]
        payments=None,  # type: ignore[arg-type]
        audit=None,  # type: ignore[arg-type]
        storage=None,
    )

    import asyncio

    result = asyncio.run(
        service.list_invoices("user-1", limit=50, offset=0)
    )

    assert isinstance(result, Success)
    payload = result.unwrap()
    assert isinstance(payload, InvoiceListResponse)
    assert payload.total == 7
    assert payload.limit == 50
    assert payload.offset == 0
    assert [str(item.id) for item in payload.items] == [str(invoice.id)]


def test_invoice_list_route_declares_the_paginated_response_model() -> None:
    route = next(route for route in invoices_router.routes if route.path == "/invoices")
    response_model = route.response_model
    assert response_model is not None
    assert response_model == APIResponse[InvoiceListResponse]


def test_router_openapi_shape_lists_paginated_fields() -> None:
    """The declared response model serializes the subscriptions' field set."""
    fields = set(InvoiceListResponse.model_fields)
    assert fields == {"items", "total", "limit", "offset"}
