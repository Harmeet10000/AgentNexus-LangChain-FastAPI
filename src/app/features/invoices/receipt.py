"""Payment receipt SQLAlchemy model (non-taxable payment acknowledgment)."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared import Base


class PaymentReceipt(Base):
    """Payment receipt - non-taxable acknowledgment of payment received."""

    __tablename__ = "payment_receipts"
    __table_args__ = (
        UniqueConstraint("receipt_number", name="uq_payment_receipts_receipt_number"),
        UniqueConstraint("payment_id", name="uq_payment_receipts_payment_id"),
        Index("ix_payment_receipts_subscription_id", "subscription_id"),
        Index("ix_payment_receipts_user_id", "user_id"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    receipt_number: Mapped[str] = mapped_column(String(length=32), nullable=False)
    subscription_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="subscriptions.id", ondelete="CASCADE"),
        nullable=False,
    )
    payment_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="payments.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)

    amount: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(length=3), nullable=False, default="INR")
    payment_method: Mapped[str | None] = mapped_column(String(length=16), nullable=True)

    razorpay_payment_id: Mapped[str] = mapped_column(String(length=64), nullable=False)

    receipt_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    billing_period_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    billing_period_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    plan_name: Mapped[str | None] = mapped_column(String(length=128), nullable=True)

    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
