"""Proration request/response DTOs."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal


class ProrationDirection(StrEnum):
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    NO_CHANGE = "no_change"


class ProrationCalculation(BaseModel):
    """Proration result for a mid-cycle plan change."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True)

    subscription_id: str = Field(serialization_alias="subscriptionId")
    current_plan_id: str = Field(serialization_alias="currentPlanId")
    new_plan_id: str = Field(serialization_alias="newPlanId")
    effective_date: datetime = Field(serialization_alias="effectiveDate")
    remaining_fraction: Decimal = Field(serialization_alias="remainingFraction")

    # Prorated value of the unused portion of the current plan, in paisa.
    current_plan_prorated: int = Field(serialization_alias="currentPlanProrated")
    # Prorated cost of the new plan for the same remaining period, in paisa.
    new_plan_prorated: int = Field(serialization_alias="newPlanProrated")
    # Difference charged (positive) or credited (negative), in paisa.
    proration_amount: int = Field(serialization_alias="prorationAmount")
    tax_amount: int = Field(default=0, serialization_alias="taxAmount")
    total_amount: int = Field(default=0, serialization_alias="totalAmount")
    direction: ProrationDirection
