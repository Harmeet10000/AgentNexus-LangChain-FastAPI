"""SQLAlchemy models for the transactional outbox."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import sqlalchemy.dialects.postgresql  # noqa: F401 — used for JSONB type reference
from sqlalchemy import DateTime, Index, Integer, String, Text, dialects, text
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class OutboxEvent(Base):
    __tablename__ = "outbox_events"
    # Partial index over the relay's only hot query — unpublished events in
    # arrival order. Declared here rather than only in revision a5bd6b69a28e so
    # the registry and the database agree; an index the database has and the
    # registry does not is one `alembic check` proposes to REMOVE.
    __table_args__ = (
        Index(
            "idx_outbox_unpublished",
            "created_at",
            postgresql_where=text("published_at IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(dialects.postgresql.JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # server_default mirrors the DDL's DEFAULT 0. The Python-side default alone
    # left the two disagreeing, which `alembic check` reports as modify_default.
    publish_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DeadLetterEvent(Base):
    __tablename__ = "dead_letter_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_event_id: Mapped[str] = mapped_column(String(36), nullable=False)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(dialects.postgresql.JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    dead_letter_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    last_error: Mapped[str] = mapped_column(Text, nullable=False)
