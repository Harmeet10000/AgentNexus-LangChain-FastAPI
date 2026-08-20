"""Immutable audit log SQLAlchemy model for compliance."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    Index,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class AuditAction(StrEnum):
    SUBSCRIPTION_CREATED = "subscription.created"
    SUBSCRIPTION_ACTIVATED = "subscription.activated"
    SUBSCRIPTION_CANCELLED = "subscription.cancelled"
    SUBSCRIPTION_HALTED = "subscription.halted"
    PLAN_CHANGED = "plan.changed"
    PAYMENT_CAPTURED = "payment.captured"
    PAYMENT_FAILED = "payment.failed"
    PAYMENT_RECEIVED = "payment.received"
    REFUND_ISSUED = "refund.issued"
    INVOICE_GENERATED = "invoice.generated"
    INVOICE_VOIDED = "invoice.voided"
    INVOICE_REISSUED = "invoice.reissued"
    TRIAL_EXTENSION_REQUESTED = "trial_extension.requested"
    TRIAL_EXTENSION_APPROVED = "trial_extension.approved"
    TRIAL_EXTENSION_REJECTED = "trial_extension.rejected"
    BATCH_INVOICE_GENERATED = "batch.invoice_generated"
    EMAIL_SENT = "email.sent"
    REPORT_GENERATED = "report.generated"
    RECONCILIATION_RUN = "reconciliation.run"
    SUBSCRIPTION_AUTHENTICATED = "subscription.authenticated"
    SUBSCRIPTION_PAUSED = "subscription.paused"
    SUBSCRIPTION_RESUMED = "subscription.resumed"
    REFUND_PROCESSED = "refund.processed"
    CHARGEBACK = "payment.dispute.created"
    WEBHOOK_REPLAYED = "webhook.replayed"
    PLAN_VERSIONED = "plan.versioned"


class AuditLog(Base):
    """Immutable audit trail. Append-only: no update/delete paths exist."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_entity_type_id", "entity_type", "entity_id"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_created_at", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    entity_type: Mapped[str] = mapped_column(String(length=32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(length=64), nullable=False)
    action: Mapped[str] = mapped_column(String(length=64), nullable=False)

    user_id: Mapped[str | None] = mapped_column(String(length=255), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(length=64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(length=512), nullable=True)

    changes: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    metadata_: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
