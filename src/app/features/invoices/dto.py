"""Invoice, receipt, and report request/response DTOs."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.features.invoices.report import ReportFormat, ReportType


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


class ReceiptResponse(BaseModel):
    """Payment receipt representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    receipt_number: str = Field(serialization_alias="receiptNumber")
    subscription_id: str = Field(serialization_alias="subscriptionId")
    payment_id: str = Field(serialization_alias="paymentId")
    user_id: str = Field(serialization_alias="userId")
    amount: Decimal
    currency: str
    payment_method: str | None = Field(default=None, serialization_alias="paymentMethod")
    razorpay_payment_id: str = Field(serialization_alias="razorpayPaymentId")
    receipt_date: datetime = Field(serialization_alias="receiptDate")
    billing_period_start: datetime | None = Field(
        default=None, serialization_alias="billingPeriodStart"
    )
    billing_period_end: datetime | None = Field(
        default=None, serialization_alias="billingPeriodEnd"
    )
    plan_name: str | None = Field(default=None, serialization_alias="planName")
    pdf_url: str | None = Field(default=None, serialization_alias="pdfUrl")


class ReportCreateDTO(BaseModel):
    """Request a report generation."""

    model_config = ConfigDict(extra="forbid")

    report_type: ReportType
    report_name: str = Field(min_length=1, max_length=200)
    date_from: datetime | None = None
    date_to: datetime | None = None
    plan_ids: list[str] | None = None
    output_format: ReportFormat = ReportFormat.CSV


class ReportResponse(BaseModel):
    """Report representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    report_type: str = Field(serialization_alias="reportType")
    report_name: str = Field(serialization_alias="reportName")
    status: str
    date_from: datetime | None = Field(default=None, serialization_alias="dateFrom")
    date_to: datetime | None = Field(default=None, serialization_alias="dateTo")
    generated_at: datetime | None = Field(default=None, serialization_alias="generatedAt")
    output_format: str = Field(serialization_alias="outputFormat")
    file_url: str | None = Field(default=None, serialization_alias="fileUrl")
    row_count: int | None = Field(default=None, serialization_alias="rowCount")
    created_at: datetime = Field(serialization_alias="createdAt")
