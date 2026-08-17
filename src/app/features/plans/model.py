"""Plan and billing-interval SQLAlchemy models."""

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
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared import Base


class BillingInterval(StrEnum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class Plan(Base):
    """Billing plan with pricing and features.

    Amounts are stored in the smallest currency unit (paisa for INR, cents
    for USD/EUR/GBP) to match the Razorpay API and avoid float drift.
    """

    __tablename__ = "plans"
    __table_args__ = (
        Index(
            "uq_plans_active_name",
            "name",
            unique=True,
            postgresql_where=text("is_active"),
        ),
        Index("ix_plans_is_active", "is_active"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    parent_plan_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, default=None
    )
    razorpay_plan_id: Mapped[str | None] = mapped_column(String(length=64), nullable=True)
    name: Mapped[str] = mapped_column(String(length=128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(length=3), nullable=False, default="INR")
    interval: Mapped[str] = mapped_column(String(length=16), nullable=False)
    interval_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    trial_period_days: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    tax_rate: Mapped[Decimal] = mapped_column(
        Numeric(8, 6), nullable=False, default=Decimal("0.18")
    )
    refund_policy: Mapped[str] = mapped_column(
        String(length=16), nullable=False, default="PRO_RATA"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    features: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    metadata_: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
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
