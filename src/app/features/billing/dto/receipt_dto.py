"""Payment receipt request/response DTOs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal


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
