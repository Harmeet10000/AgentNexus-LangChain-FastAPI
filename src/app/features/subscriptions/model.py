"""Subscription and subscription-status SQLAlchemy models."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared import Base


class SubscriptionStatus(StrEnum):
    CREATED = "created"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    HALTED = "halted"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    EXPIRED = "expired"


class Subscription(Base):
    """User subscription to a billing plan.

    ``version`` is the optimistic-locking field: every update runs
    ``UPDATE ... WHERE id = :id AND version = :expected`` and bumps the
    version, so concurrent writers cannot silently lose updates.
    """

    __tablename__ = "subscriptions"
    __table_args__ = (
        Index("ix_subscriptions_user_id", "user_id"),
        Index("ix_subscriptions_razorpay_subscription_id", "razorpay_subscription_id"),
        Index("ix_subscriptions_plan_id", "plan_id"),
        Index("ix_subscriptions_id_version", "id", "version"),
        Index(
            "uq_subscriptions_user_plan_active",
            "user_id",
            "plan_id",
            unique=True,
            postgresql_where=text(
                "deleted_at IS NULL AND status NOT IN ('cancelled', 'expired')"
            ),
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    plan_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(column="plans.id"), nullable=False
    )
    razorpay_subscription_id: Mapped[str | None] = mapped_column(String(length=64), nullable=True)
    razorpay_customer_id: Mapped[str | None] = mapped_column(String(length=64), nullable=True)

    status: Mapped[str] = mapped_column(String(length=16), nullable=False, default="created")

    current_period_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    current_period_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    trial_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pause_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pause_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    retry_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(BigInteger, nullable=False, default=4)

    currency_display: Mapped[str] = mapped_column(String(length=3), nullable=False, default="INR")
    trial_extension_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    version: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

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
