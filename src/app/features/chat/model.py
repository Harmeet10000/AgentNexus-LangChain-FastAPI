"""Chat session/message SQLAlchemy models — feature-owned, not database/schemas.

Canonical location for chat persistence. `database/schemas/chat_messages.py` remains
as a shim re-exporting these for one release (import path deprecation).
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Enum, Integer, String, Text
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from database.base import Base


class ChatSession(Base):
    """Store chat session metadata — one row per conversation."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Legacy: Python-side defaults match 0014 DB (no server_default there).
    # New tables should use server_default=func.now() (see database/base.py).
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class ChatMessage(Base):
    """Store chat messages between user and LLM."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # Native PG enum message_role — created by 0014 DO block; keep name stable.
    role: Mapped[str] = mapped_column(
        Enum("user", "assistant", "system", name="message_role"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, default=lambda: datetime.now(UTC)
    )
