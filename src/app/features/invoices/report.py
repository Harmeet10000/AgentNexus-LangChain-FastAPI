"""Report SQLAlchemy model for analytics and export."""

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


class ReportType(StrEnum):
    REVENUE = "revenue"
    CHURN = "churn"
    AR_AGING = "ar_aging"
    TAX_LIABILITY = "tax_liability"
    SUBSCRIPTION_SUMMARY = "subscription_summary"
    PAYMENT_SUMMARY = "payment_summary"


class ReportFormat(StrEnum):
    PDF = "pdf"
    CSV = "csv"
    XLSX = "xlsx"


class ReportStatus(StrEnum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class Report(Base):
    """Generated report for analytics and export."""

    __tablename__ = "reports"
    __table_args__ = (
        Index("ix_reports_report_type", "report_type"),
        Index("ix_reports_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    report_type: Mapped[str] = mapped_column(String(length=32), nullable=False)
    report_name: Mapped[str] = mapped_column(String(length=200), nullable=False)

    status: Mapped[str] = mapped_column(String(length=16), nullable=False, default="pending")

    date_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    date_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    plan_ids: Mapped[list[object] | None] = mapped_column(JSONB, nullable=True)
    user_ids: Mapped[list[object] | None] = mapped_column(JSONB, nullable=True)

    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    generated_by_user_id: Mapped[str | None] = mapped_column(String(length=255), nullable=True)

    output_format: Mapped[str] = mapped_column(String(length=8), nullable=False, default="csv")
    file_url: Mapped[str | None] = mapped_column(String(length=1000), nullable=True)
    row_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

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
