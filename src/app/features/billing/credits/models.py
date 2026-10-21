# ── from consumption.py ──
"""CreditConsumption SQLAlchemy model."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class CreditConsumption(Base):
    """Credit consumption ledger record.

    Tracks when and how much credit was applied to an invoice.
    ``consumed_amount`` is stored in paisa.
    """

    __tablename__ = "credit_consumptions"
    __table_args__ = (
        Index("ix_credit_consumptions_user_id", "user_id"),
        Index("ix_credit_consumptions_credit_id", "credit_id"),
        Index("ix_credit_consumptions_invoice_id", "invoice_id"),
        Index("ix_credit_consumptions_created_at", "created_at"),
        # CHECK constraint for data integrity
        sa.CheckConstraint(
            "consumed_amount > 0", name="ck_credit_consumptions_consumed_amount_positive"
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    credit_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="user_credits.id", ondelete="RESTRICT"),
        nullable=False,
    )
    invoice_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="invoices.id"),
        nullable=True,
    )
    razorpay_payment_id: Mapped[str | None] = mapped_column(String(length=64), nullable=True)

    consumed_amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )


# ── from credit.py ──
"""UserCredit SQLAlchemy model and enums."""


class CreditType(StrEnum):
    """Origin of the credit grant."""

    PLAN_CREDIT = "plan_credit"  # From plan downgrade proration
    PROMOTIONAL = "promotional"  # Marketing/promotional credit
    ADMIN_GRANT = "admin_grant"  # Admin-granted goodwill credit


class CreditStatus(StrEnum):
    """Lifecycle status of a credit record."""

    ACTIVE = "active"  # Available for consumption
    CONSUMED = "consumed"  # Fully consumed
    EXPIRED = "expired"  # Past valid_until timestamp


class CreditNoteStatus(StrEnum):
    PENDING = "pending"
    AVAILABLE = "available"
    APPLIED = "applied"
    CANCELLED = "cancelled"


class UserCredit(Base):
    """User credit balance record.

    ``credit_amount`` and ``remaining_balance`` are stored in paisa (smallest
    currency unit) to match Payment.amount convention.
    """

    __tablename__ = "user_credits"
    __table_args__ = (
        Index("ix_user_credits_user_id", "user_id"),
        Index("ix_user_credits_status", "status"),
        Index("ix_user_credits_valid_until", "valid_until"),
        Index("ix_user_credits_created_at", "created_at"),
        # Expression index for the idempotency lookup in
        # credits/repository.find_credit_by_operation_key (GIN(metadata_)
        # cannot serve `->>` equality).
        Index("ix_user_credits_operation_key", text("(metadata_ ->> 'operation_key')")),
        # CHECK constraints for data integrity
        sa.CheckConstraint("credit_amount > 0", name="ck_user_credits_credit_amount_positive"),
        sa.CheckConstraint(
            "remaining_balance >= 0 AND remaining_balance <= credit_amount",
            name="ck_user_credits_remaining_balance_valid",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'consumed', 'expired')",
            name="ck_user_credits_status",
        ),
        sa.CheckConstraint(
            "valid_until IS NULL OR valid_from <= valid_until",
            name="ck_user_credits_validity_range",
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)

    # Credit details
    credit_type: Mapped[str] = mapped_column(String(length=32), nullable=False)
    credit_amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    remaining_balance: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # Validity period
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(tz=UTC),
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    # Status tracking
    status: Mapped[str] = mapped_column(String(length=16), nullable=False, default="active")
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Metadata for audit and tracking
    metadata_: Mapped[dict[str, object]] = mapped_column(
        MutableDict.as_mutable(JSONB), default=dict, nullable=False
    )

    # Soft delete for audit compliance
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

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


class CreditNote(Base):
    """Durable downgrade compensation awaiting the next renewal invoice."""

    __tablename__ = "credit_notes"
    __table_args__ = (
        UniqueConstraint("operation_key", name="uq_credit_notes_operation_key"),
        Index("ix_credit_notes_subscription_id", "subscription_id"),
        Index("ix_credit_notes_credit_id", "credit_id"),
        Index("ix_credit_notes_source_invoice_id", "source_invoice_id"),
        Index("ix_credit_notes_applies_at", "applies_at"),
        sa.CheckConstraint("amount_paisa > 0", name="ck_credit_notes_amount_positive"),
        sa.CheckConstraint(
            "remaining_paisa >= 0 AND remaining_paisa <= amount_paisa",
            name="ck_credit_notes_remaining_valid",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'available', 'applied', 'cancelled')",
            name="ck_credit_notes_status",
        ),
        # Existing rows created before the downgrade source invariant may be
        # NULL.  Migration 0026 marks those rows for manual review; every new
        # note must carry a source invoice unless it is explicitly retained in
        # that review state.  Keeping the column nullable is intentional until
        # historical review is complete.
        sa.CheckConstraint(
            "source_invoice_id IS NOT NULL OR (metadata_->>'manual_review') = 'true'",
            name="ck_credit_notes_source_invoice_required",
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    subscription_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False
    )
    # A note is the authoritative economic ledger for downgrade credit.  The
    # user-credit row is a projection retained for the existing credit API and
    # consumption FK compatibility.
    credit_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("user_credits.id", ondelete="RESTRICT"), nullable=True
    )
    source_invoice_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("invoices.id", ondelete="RESTRICT"), nullable=True
    )
    operation_key: Mapped[str] = mapped_column(String(length=128), nullable=False)
    amount_paisa: Mapped[int] = mapped_column(BigInteger, nullable=False)
    remaining_paisa: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(length=3), nullable=False)
    applies_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(length=16), nullable=False, default="available")
    metadata_: Mapped[dict[str, object]] = mapped_column(
        MutableDict.as_mutable(JSONB), default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(tz=UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
        nullable=False,
    )


class CreditNoteApplication(Base):
    """Immutable per-invoice application of an authoritative credit note.

    The unique pair is the database idempotency boundary.  Application rows
    remain even after a note reaches zero so webhook retries cannot spend it a
    second time.
    """

    __tablename__ = "credit_note_applications"
    __table_args__ = (
        UniqueConstraint(
            "credit_note_id",
            "invoice_id",
            name="uq_credit_note_applications_note_invoice",
        ),
        Index("ix_credit_note_applications_invoice_id", "invoice_id"),
        Index("ix_credit_note_applications_credit_note_id", "credit_note_id"),
        sa.CheckConstraint("amount_paisa > 0", name="ck_credit_note_applications_amount_positive"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    credit_note_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("credit_notes.id", ondelete="RESTRICT"),
        nullable=False,
    )
    invoice_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("invoices.id", ondelete="RESTRICT"), nullable=False
    )
    amount_paisa: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(tz=UTC), nullable=False
    )
