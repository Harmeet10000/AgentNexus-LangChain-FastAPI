"""CreditConsumption SQLAlchemy model."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared import Base


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
