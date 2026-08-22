"""UserCredit SQLAlchemy model and enums."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


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
        Index(
            "uq_user_credits_user_id_active",
            "user_id",
            unique=True,
            postgresql_where=text("status = 'active' AND deleted_at IS NULL"),
        ),
        # CHECK constraints for data integrity
        sa.CheckConstraint("credit_amount > 0", name="ck_user_credits_credit_amount_positive"),
        sa.CheckConstraint(
            "remaining_balance <= credit_amount",
            name="ck_user_credits_remaining_balance_lte_amount",
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
    metadata_: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)

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
