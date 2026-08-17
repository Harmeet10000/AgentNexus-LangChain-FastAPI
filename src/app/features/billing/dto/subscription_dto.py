"""Subscription request/response DTOs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any


class SubscriptionCreateDTO(BaseModel):
    """Create a subscription for the current user."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(description="UUID of an active plan")
    customer_email: str = Field(min_length=3, max_length=320)
    customer_phone: str | None = Field(default=None, pattern=r"^[0-9]{10,15}$")
    customer_notify: bool = True
    trial_period_days: int | None = Field(default=None, ge=1, le=365)


class PlanChangeDTO(BaseModel):
    """Change the plan of an active subscription."""

    model_config = ConfigDict(extra="forbid")

    new_plan_id: str
    effective_date: datetime | None = None


class SubscriptionCancelDTO(BaseModel):
    """Cancel a subscription, immediately or at period end."""

    model_config = ConfigDict(extra="forbid")

    cancel_at_period_end: bool = True
    reason: str | None = Field(default=None, max_length=500)


class SubscriptionPauseDTO(BaseModel):
    """Pause a subscription, optionally for a fixed duration."""

    model_config = ConfigDict(extra="forbid")

    pause_duration_days: int | None = Field(default=None, ge=1, le=365)


class SubscriptionResponse(BaseModel):
    """Subscription representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    user_id: str = Field(serialization_alias="userId")
    plan_id: str = Field(serialization_alias="planId")
    plan: dict[str, Any] | None = None
    razorpay_subscription_id: str | None = Field(
        default=None, serialization_alias="razorpaySubscriptionId"
    )
    status: str
    current_period_start: datetime | None = Field(
        default=None, serialization_alias="currentPeriodStart"
    )
    current_period_end: datetime | None = Field(
        default=None, serialization_alias="currentPeriodEnd"
    )
    trial_end: datetime | None = Field(default=None, serialization_alias="trialEnd")
    cancel_at_period_end: bool = Field(default=False, serialization_alias="cancelAtPeriodEnd")
    pause_start: datetime | None = Field(default=None, serialization_alias="pauseStart")
    pause_end: datetime | None = Field(default=None, serialization_alias="pauseEnd")
    retry_count: int = Field(default=0, serialization_alias="retryCount")
    currency_display: str = Field(default="INR", serialization_alias="currencyDisplay")
    version: int = Field(default=0)
    payment_url: str | None = Field(default=None, serialization_alias="paymentUrl")
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")


class SubscriptionListResponse(BaseModel):
    """Paginated list of subscriptions."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    items: list[SubscriptionResponse]
    total: int
    limit: int
    offset: int
