"""Invoice, invoice-line-item, and invoice-status SQLAlchemy models."""

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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


class InvoiceStatus(StrEnum):
    DRAFT = "draft"
    ISSUED = "issued"
    PAID = "paid"
    VOID = "void"


class Invoice(Base):
    """GST-compliant invoice.

    GST-inclusive pricing: ``total = subtotal + tax`` where
    ``subtotal = amount / 1.18`` and ``tax = subtotal * 0.18``, all computed
    in integer paisa so ``total * 100 == payment.amount`` exactly.
    """

    __tablename__ = "invoices"
    __table_args__ = (
        UniqueConstraint("invoice_number", name="uq_invoices_invoice_number"),
        Index("ix_invoices_subscription_id", "subscription_id"),
        Index("ix_invoices_payment_id", "payment_id"),
        Index("ix_invoices_user_id", "user_id"),
        Index("ix_invoices_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    invoice_number: Mapped[str] = mapped_column(String(length=32), nullable=False)
    subscription_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="subscriptions.id", ondelete="CASCADE"),
        nullable=False,
    )
    payment_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey(column="payments.id"), nullable=True
    )
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)

    status: Mapped[str] = mapped_column(String(length=16), nullable=False, default="draft")

    subtotal: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    tax_rate: Mapped[Decimal] = mapped_column(
        Numeric(8, 6), nullable=False, default=Decimal("0.18")
    )
    tax_amount: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    total: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(length=3), nullable=False, default="INR")

    seller_gstin: Mapped[str] = mapped_column(String(length=15), nullable=False)
    buyer_gstin: Mapped[str | None] = mapped_column(String(length=15), nullable=True)
    place_of_supply: Mapped[str] = mapped_column(String(length=2), nullable=False)
    sac_code: Mapped[str] = mapped_column(String(length=6), nullable=False, default="998314")

    cgst_amount: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False, default=Decimal(0))
    sgst_amount: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False, default=Decimal(0))
    igst_amount: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False, default=Decimal(0))

    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pdf_url: Mapped[str | None] = mapped_column(Text, nullable=True)

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

    line_items: Mapped[list["InvoiceLineItem"]] = relationship(
        back_populates="invoice",
        cascade="all, delete-orphan",
    )


class InvoiceLineItem(Base):
    """Single line item on an invoice (Requirement 39)."""

    __tablename__ = "invoice_line_items"
    __table_args__ = (Index("ix_invoice_line_items_invoice_id", "invoice_id"),)

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    invoice_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="invoices.id", ondelete="CASCADE"),
        nullable=False,
    )
    plan_name: Mapped[str] = mapped_column(String(length=128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    quantity: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False)
    tax_amount: Mapped[Decimal] = mapped_column(Numeric(20, 2), nullable=False, default=Decimal(0))
    sac_code: Mapped[str] = mapped_column(String(length=6), nullable=False, default="998314")

    invoice: Mapped[Invoice] = relationship(back_populates="line_items")
