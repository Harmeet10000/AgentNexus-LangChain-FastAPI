"""Currency and FX-rate SQLAlchemy models."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared import Base


class CurrencyCode(StrEnum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    AUD = "AUD"
    CAD = "CAD"
    JPY = "JPY"
    SGD = "SGD"
    CHF = "CHF"
    CNY = "CNY"


class Currency(Base):
    """Currency configuration with ISO 4217 code."""

    __tablename__ = "currencies"
    __table_args__ = (UniqueConstraint("code", name="uq_currencies_code"),)

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    code: Mapped[str] = mapped_column(String(length=3), nullable=False)
    name: Mapped[str] = mapped_column(String(length=64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(length=8), nullable=False)
    iso_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    decimal_places: Mapped[int] = mapped_column(BigInteger, nullable=False, default=2)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
        nullable=False,
    )


class FXRateSource(StrEnum):
    RAZORPAY = "razorpay"
    RBI = "rbi"
    ECB = "ecb"
    MANUAL = "manual"


class FXRate(Base):
    """Foreign exchange rate for currency conversion."""

    __tablename__ = "fx_rates"
    __table_args__ = (
        UniqueConstraint(
            "base_currency", "target_currency", "effective_at", name="uq_fx_rates_pair_period"
        ),
        Index("ix_fx_rates_pair", "base_currency", "target_currency"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    base_currency: Mapped[str] = mapped_column(String(length=3), nullable=False)
    target_currency: Mapped[str] = mapped_column(String(length=3), nullable=False)

    rate: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    source: Mapped[str] = mapped_column(String(length=16), nullable=False, default="razorpay")

    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    manually_entered_by_user_id: Mapped[str | None] = mapped_column(
        String(length=255), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
