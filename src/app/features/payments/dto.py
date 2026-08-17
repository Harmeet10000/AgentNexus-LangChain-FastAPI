"""Payment request/response DTOs."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime

    from app.features.payments.currency import CurrencyCode
    from app.features.payments.model import PaymentMethod


class PaymentRecordDTO(BaseModel):
    """Payment details extracted from a payment.captured webhook.

    ``amount`` is in paisa to match the Razorpay payload.
    """

    model_config = ConfigDict(extra="forbid")

    razorpay_payment_id: str
    subscription_id: str
    amount: int = Field(gt=0)
    currency: str = Field(default="INR")
    method: PaymentMethod | None = None
    razorpay_order_id: str | None = None
    captured_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict, serialization_alias="metadata")


class PaymentResponse(BaseModel):
    """Payment representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    subscription_id: str = Field(serialization_alias="subscriptionId")
    invoice_id: str | None = Field(default=None, serialization_alias="invoiceId")
    razorpay_payment_id: str = Field(serialization_alias="razorpayPaymentId")
    amount: int
    currency: str
    status: str
    method: str | None = None
    captured_at: datetime | None = Field(default=None, serialization_alias="capturedAt")
    failed_at: datetime | None = Field(default=None, serialization_alias="failedAt")
    error_code: str | None = Field(default=None, serialization_alias="errorCode")
    error_description: str | None = Field(default=None, serialization_alias="errorDescription")
    refund_amount: Decimal = Field(default=Decimal(0), serialization_alias="refundAmount")
    created_at: datetime = Field(serialization_alias="createdAt")


class RefundRequestDTO(BaseModel):
    """Initiate a full or partial refund. ``amount`` is in paisa."""

    model_config = ConfigDict(extra="forbid")

    amount: int = Field(gt=0, description="Refund amount in paisa")
    reason: str | None = Field(default=None, max_length=500)
    notes: dict[str, object] = Field(default_factory=dict)


class RefundResponse(BaseModel):
    """Refund representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    razorpay_refund_id: str = Field(serialization_alias="razorpayRefundId")
    payment_id: str = Field(serialization_alias="paymentId")
    amount: int
    currency: str
    status: str
    created_at: datetime = Field(serialization_alias="createdAt")


class CurrencyResponse(BaseModel):
    """Currency configuration representation."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    code: str
    name: str
    symbol: str
    iso_number: int = Field(serialization_alias="isoNumber")
    decimal_places: int = Field(serialization_alias="decimalPlaces")
    is_active: bool = Field(serialization_alias="isActive")


class FXRateCreateDTO(BaseModel):
    """Manually set an FX rate."""

    model_config = ConfigDict(extra="forbid")

    base_currency: CurrencyCode
    target_currency: CurrencyCode
    rate: Decimal = Field(gt=0)
    effective_at: datetime | None = None


class FXRateResponse(BaseModel):
    """FX rate representation."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    base_currency: str = Field(serialization_alias="baseCurrency")
    target_currency: str = Field(serialization_alias="targetCurrency")
    rate: Decimal
    source: str
    effective_at: datetime = Field(serialization_alias="effectiveAt")
    expires_at: datetime | None = Field(default=None, serialization_alias="expiresAt")
