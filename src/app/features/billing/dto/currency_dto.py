"""Currency and FX-rate request/response DTOs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal

    from app.features.billing.models.currency import CurrencyCode


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
