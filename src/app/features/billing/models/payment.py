"""Payment and payment-status/method SQLAlchemy models."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared import Base


class PaymentStatus(StrEnum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class PaymentMethod(StrEnum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMI = "emi"


class Payment(Base):
    """Payment transaction record.

    ``amount`` is stored in paisa (smallest currency unit) so that
    ``invoice.total * 100 == payment.amount`` holds exactly (Property 1).
    """

    __tablename__ = "payments"
    __table_args__ = (
        UniqueConstraint("razorpay_payment_id", name="uq_payments_razorpay_payment_id"),
        Index("ix_payments_subscription_id", "subscription_id"),
        Index("ix_payments_invoice_id", "invoice_id"),
        Index("ix_payments_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    subscription_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="subscriptions.id", ondelete="CASCADE"),
        nullable=False,
    )
    invoice_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="invoices.id"),
        nullable=True,
    )

    razorpay_payment_id: Mapped[str] = mapped_column(String(length=64), nullable=False)
    razorpay_order_id: Mapped[str | None] = mapped_column(String(length=64), nullable=True)

    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(length=3), nullable=False, default="INR")
    status: Mapped[str] = mapped_column(String(length=24), nullable=False)

    method: Mapped[str | None] = mapped_column(String(length=16), nullable=True)

    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(length=64), nullable=True)
    error_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    refund_amount: Mapped[Decimal] = mapped_column(
        Numeric(20, 2), nullable=False, default=Decimal(0)
    )

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
