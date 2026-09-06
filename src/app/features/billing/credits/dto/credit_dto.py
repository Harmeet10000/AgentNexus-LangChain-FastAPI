"""DTOs for credit grant operations."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.utils.exceptions import ValidationException


class CreditGrantDTO(BaseModel):
    """Request to grant credit to a user.

    All amount fields are stored in paisa (smallest currency unit).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    credit_type: str = Field(
        description="Origin of the credit: plan_credit, promotional, or admin_grant"
    )
    credit_amount: int = Field(
        ge=1,
        description="Amount in paisa (minimum 1 paisa, positive integer)",
    )
    valid_from: datetime | None = Field(
        default=None,
        description="Earliest date credit is available (default: now)",
    )
    valid_until: datetime | None = Field(
        default=None,
        description="Expiry date (nullable for credits with no expiry)",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Optional description for audit trail",
    )
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific metadata (e.g., admin_user_id for admin grants)",
    )

    @field_validator("credit_amount")
    @classmethod
    def validate_credit_amount(cls, v: int) -> int:
        """Validate credit_amount is at least 1 paisa."""
        if v < 1:
            raise ValidationException(
                detail="Credit amount must be at least 1 paisa",
                error_code="CREDIT_AMOUNT_MUST_BE_POSITIVE",
            )
        return v

    @field_validator("valid_until")
    @classmethod
    def validate_date_range(cls, v: datetime | None, info) -> datetime | None:
        """Validate valid_from <= valid_until (if both set)."""
        valid_from = info.data.get("valid_from")
        if valid_from is not None and v is not None and v < valid_from:
            raise ValidationException(
                detail="valid_until cannot be earlier than valid_from",
                error_code="CREDIT_INVALID_DATE_RANGE",
            )
        return v

    @field_validator("metadata_")
    @classmethod
    def validate_admin_grant_metadata(cls, v: dict[str, Any], info) -> dict[str, Any]:
        """Validate ADMIN_GRANT has admin_user_id in metadata."""
        credit_type = info.data.get("credit_type")
        if credit_type == "admin_grant" and "admin_user_id" not in v:
            raise ValidationException(
                detail="ADMIN_GRANT requires admin_user_id in metadata",
                error_code="CREDIT_METADATA_MISSING",
            )
        return v


class CreditGrantResponse(BaseModel):
    """Response from credit grant operation."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    credit_id: str = Field(serialization_alias="creditId")
    user_id: str = Field(serialization_alias="userId")
    credit_type: str = Field(serialization_alias="creditType")
    credit_amount: int = Field(serialization_alias="creditAmount")
    remaining_balance: int = Field(serialization_alias="remainingBalance")
    valid_from: datetime = Field(serialization_alias="validFrom")
    valid_until: datetime | None = Field(default=None, serialization_alias="validUntil")
    status: str = Field(serialization_alias="status")
    created_at: datetime = Field(serialization_alias="createdAt")


class CreditBalanceResponse(BaseModel):
    """User's available credit balance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_balance: int = Field(
        serialization_alias="totalBalance",
        description="Sum of active, non-expired credits in paisa",
    )
    total_balance_rupees: float = Field(
        serialization_alias="totalBalanceRupees",
        description="Sum of active, non-expired credits in rupees (for convenience)",
    )
    currency: str = Field(default="INR")


class CreditHistoryResponse(BaseModel):
    """User's credit and consumption history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    credits: list["CreditRecord"] = Field(default_factory=list)
    consumptions: list["ConsumptionRecord"] = Field(default_factory=list)
    total: int = Field(description="Total number of records")
    limit: int = Field(description="Page size")
    offset: int = Field(description="Page offset")


class CreditRecord(BaseModel):
    """Individual credit record."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    credit_id: str = Field(serialization_alias="creditId")
    credit_type: str = Field(serialization_alias="creditType")
    credit_amount: int = Field(serialization_alias="creditAmount")
    remaining_balance: int = Field(serialization_alias="remainingBalance")
    valid_from: datetime = Field(serialization_alias="validFrom")
    valid_until: datetime | None = Field(default=None, serialization_alias="validUntil")
    status: str = Field(serialization_alias="status")
    created_at: datetime = Field(serialization_alias="createdAt")


class ConsumptionRecord(BaseModel):
    """Individual consumption record."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    consumption_id: str = Field(serialization_alias="consumptionId")
    credit_id: str = Field(serialization_alias="creditId")
    consumed_amount: int = Field(serialization_alias="consumedAmount")
    invoice_id: str | None = Field(default=None, serialization_alias="invoiceId")
    razorpay_payment_id: str | None = Field(default=None, serialization_alias="razorpayPaymentId")
    created_at: datetime = Field(serialization_alias="createdAt")
