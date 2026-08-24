"""Plan request/response DTOs."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.features.plans.model import BillingInterval


class PlanCreateDTO(BaseModel):
    """Create a billing plan. ``amount`` is in paisa (smallest currency unit)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    amount: int = Field(ge=100, description="Plan price in paisa (>= 100 = INR 1.00)")
    currency: str = Field(default="INR", pattern="^(INR|USD|EUR|GBP|AUD|CAD|JPY|SGD|CHF|CNY)$")
    interval: BillingInterval
    interval_count: int = Field(default=1, ge=1)
    trial_period_days: int = Field(default=0, ge=0)
    tax_rate: Decimal = Field(default=Decimal("0.18"))
    refund_policy: str = Field(default="PRO_RATA", pattern="^(FULL|PRO_RATA|NONE)$")
    features: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict, serialization_alias="metadata")


class PlanUpdateDTO(BaseModel):
    """Update a plan. Applied to new plan versions only (Requirement 24)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = None
    amount: int | None = Field(default=None, ge=100)
    currency: str | None = Field(
        default=None, pattern="^(INR|USD|EUR|GBP|AUD|CAD|JPY|SGD|CHF|CNY)$"
    )
    interval: BillingInterval | None = None
    interval_count: int | None = Field(default=None, ge=1)
    trial_period_days: int | None = Field(default=None, ge=0)
    tax_rate: Decimal | None = None
    refund_policy: str | None = Field(default=None, pattern="^(FULL|PRO_RATA|NONE)$")
    features: dict[str, object] | None = None
    metadata: dict[str, object] | None = Field(default=None, serialization_alias="metadata")


class PlanResponse(BaseModel):
    """Plan representation returned by the API."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    id: str
    parent_plan_id: str | None = Field(default=None, serialization_alias="parentPlanId")
    razorpay_plan_id: str | None = Field(default=None, serialization_alias="razorpayPlanId")
    name: str
    description: str | None = None
    amount: int
    currency: str
    interval: str
    interval_count: int = Field(serialization_alias="intervalCount")
    trial_period_days: int = Field(serialization_alias="trialPeriodDays")
    tax_rate: Decimal = Field(serialization_alias="taxRate")
    refund_policy: str = Field(serialization_alias="refundPolicy")
    is_active: bool = Field(serialization_alias="isActive")
    features: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict, serialization_alias="metadata")
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")
