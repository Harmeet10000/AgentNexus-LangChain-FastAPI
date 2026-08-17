"""Invoice request/response DTOs."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime


class InvoiceLineItemDTO(BaseModel):
    """Single line item on an invoice."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    plan_name: str = Field(serialization_alias="planName")
    description: str | None = None
    quantity: int
    unit_price: Decimal = Field(serialization_alias="unitPrice")
    amount: Decimal
    tax_amount: Decimal = Field(default=Decimal(0), serialization_alias="taxAmount")
    sac_code: str = Field(default="998314", serialization_alias="sacCode")


class InvoiceResponse(BaseModel):
    """Invoice representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    invoice_number: str = Field(serialization_alias="invoiceNumber")
    subscription_id: str = Field(serialization_alias="subscriptionId")
    payment_id: str | None = Field(default=None, serialization_alias="paymentId")
    status: str
    subtotal: Decimal
    tax_rate: Decimal = Field(serialization_alias="taxRate")
    tax_amount: Decimal = Field(serialization_alias="taxAmount")
    total: Decimal
    currency: str
    seller_gstin: str = Field(serialization_alias="sellerGstin")
    buyer_gstin: str | None = Field(default=None, serialization_alias="buyerGstin")
    place_of_supply: str = Field(serialization_alias="placeOfSupply")
    sac_code: str = Field(serialization_alias="sacCode")
    cgst_amount: Decimal = Field(default=Decimal(0), serialization_alias="cgstAmount")
    sgst_amount: Decimal = Field(default=Decimal(0), serialization_alias="sgstAmount")
    igst_amount: Decimal = Field(default=Decimal(0), serialization_alias="igstAmount")
    issued_at: datetime | None = Field(default=None, serialization_alias="issuedAt")
    due_at: datetime | None = Field(default=None, serialization_alias="dueAt")
    paid_at: datetime | None = Field(default=None, serialization_alias="paidAt")
    pdf_url: str | None = Field(default=None, serialization_alias="pdfUrl")
    line_items: list[InvoiceLineItemDTO] = Field(
        default_factory=list, serialization_alias="lineItems"
    )
    created_at: datetime = Field(serialization_alias="createdAt")


class CreditNoteResponse(BaseModel):
    """Credit note representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    invoice_id: str = Field(serialization_alias="invoiceId")
    amount: Decimal
    reason: str
    created_at: datetime = Field(serialization_alias="createdAt")


class VoidInvoiceDTO(BaseModel):
    """Void an issued invoice, optionally reissuing it."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(description="VoidReason enum value")
    description: str | None = None
    reissue: bool = False
