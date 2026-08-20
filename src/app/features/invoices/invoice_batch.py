"""Invoice batch SQLAlchemy model for batch generation tracking."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class BatchStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class InvoiceBatch(Base):
    """Batch invoice generation record for tracking."""

    __tablename__ = "invoice_batches"
    __table_args__ = (Index("ix_invoice_batches_status", "status"),)

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    batch_name: Mapped[str] = mapped_column(String(length=200), nullable=False)

    status: Mapped[str] = mapped_column(String(length=16), nullable=False, default="pending")

    plan_ids: Mapped[list[object] | None] = mapped_column(JSONB, nullable=True)
    status_filter: Mapped[list[object] | None] = mapped_column(JSONB, nullable=True)
    date_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    date_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    total_subscriptions: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    invoices_generated: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    failed_subscriptions: Mapped[list[object]] = mapped_column(JSONB, default=list, nullable=False)

    initiated_by_user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    initiated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pdf_urls: Mapped[list[object]] = mapped_column(JSONB, default=list, nullable=False)

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
