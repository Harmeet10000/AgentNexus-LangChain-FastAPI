"""Email template SQLAlchemy model."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.shared import Base


class EmailType(StrEnum):
    WELCOME = "welcome"
    RENEWAL_REMINDER = "renewal_reminder"
    PAYMENT_FAILED = "payment_failed"
    PAYMENT_SUCCESS = "payment_success"
    CANCELLATION = "cancellation"
    REFUND_ISSUED = "refund_issued"
    SUBSCRIPTION_PAUSED = "subscription_paused"
    SUBSCRIPTION_RESUMED = "subscription_resumed"


class EmailTemplate(Base):
    """Email template for automated billing notifications."""

    __tablename__ = "email_templates"
    __table_args__ = (UniqueConstraint("email_type", name="uq_email_templates_email_type"),)

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email_type: Mapped[str] = mapped_column(String(length=32), nullable=False)

    subject: Mapped[str] = mapped_column(String(length=200), nullable=False)
    body_html: Mapped[str] = mapped_column(Text, nullable=False)
    body_plain: Mapped[str] = mapped_column(Text, nullable=False)

    variables: Mapped[list[object]] = mapped_column(JSONB, default=list, nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
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
