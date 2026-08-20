"""Invoice void record for audit trail and reissue tracking."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class VoidReason(StrEnum):
    INCORRECT_AMOUNT = "incorrect_amount"
    INCORRECT_TAX = "incorrect_tax"
    DUPLICATE_INVOICE = "duplicate_invoice"
    CUSTOMER_REQUEST = "customer_request"
    PLAN_CHANGED = "plan_changed"
    OTHER = "other"


class InvoiceVoid(Base):
    """Append-only snapshot of an invoice at the time it was voided."""

    __tablename__ = "invoice_voids"
    __table_args__ = (
        UniqueConstraint("original_invoice_id", name="uq_invoice_voids_original_invoice"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    original_invoice_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="invoices.id", ondelete="CASCADE"),
        nullable=False,
    )
    void_reason: Mapped[str] = mapped_column(String(length=32), nullable=False)
    void_description: Mapped[str | None] = mapped_column(String(length=500), nullable=True)

    voided_by_user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    voided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    original_invoice_number: Mapped[str] = mapped_column(String(length=32), nullable=False)
    original_subtotal: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    original_tax_rate: Mapped[Decimal] = mapped_column(Numeric(8, 6), nullable=False)
    original_tax_amount: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    original_total: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    original_currency: Mapped[str] = mapped_column(String(length=3), nullable=False, default="INR")

    reissued_invoice_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(column="invoices.id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
