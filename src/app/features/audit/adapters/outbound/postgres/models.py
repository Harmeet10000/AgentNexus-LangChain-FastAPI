"""PostgreSQL persistence model for the audit trail."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


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
