"""Webhook event SQLAlchemy model for idempotency and replay."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared import Base


class WebhookEventType(StrEnum):
    SUBSCRIPTION_AUTHENTICATED = "subscription.authenticated"
    SUBSCRIPTION_ACTIVATED = "subscription.activated"
    SUBSCRIPTION_CHARGED = "subscription.charged"
    SUBSCRIPTION_PENDING = "subscription.pending"
    SUBSCRIPTION_HALTED = "subscription.halted"
    SUBSCRIPTION_CANCELLED = "subscription.cancelled"
    SUBSCRIPTION_PAUSED = "subscription.paused"
    SUBSCRIPTION_RESUMED = "subscription.resumed"
    PAYMENT_AUTHORIZED = "payment.authorized"
    PAYMENT_CAPTURED = "payment.captured"
    PAYMENT_FAILED = "payment.failed"
    REFUND_CREATED = "refund.created"
    REFUND_PROCESSED = "refund.processed"
    DISPUTE_CREATED = "payment.dispute.created"


class WebhookEventStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WebhookEvent(Base):
    """Webhook event log for idempotency and replay."""

    __tablename__ = "webhook_events"
    __table_args__ = (
        UniqueConstraint("razorpay_event_id", name="uq_webhook_events_razorpay_event_id"),
        Index("ix_webhook_events_event_type", "event_type"),
        Index("ix_webhook_events_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    razorpay_event_id: Mapped[str] = mapped_column(String(length=64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(length=64), nullable=False)
    status: Mapped[str] = mapped_column(String(length=16), nullable=False, default="pending")

    payload: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)

    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(length=1000), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

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
