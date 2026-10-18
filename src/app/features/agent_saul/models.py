"""Chat session/message SQLAlchemy models — feature-owned, not database/schemas.

Canonical location for chat persistence. `database/schemas/chat_messages.py` remains
as a shim re-exporting these for one release (import path deprecation).

These models are the autogenerate comparand for the chat tables (0014 + 0022);
understanding this file as plain CRUD plumbing misses its second job: its column
types, server defaults, and constraint names are the mirror the migration must
exactly match, or `alembic check` starts proposing destructive chat-table ops.
"""

from __future__ import annotations

# Runtime-required by SQLAlchemy's mapper annotation resolution: these names
# must live in module globals when ORM annotations are evaluated (the same
# rationale as database/base.py's TC002 noqa) — do NOT move to TYPE_CHECKING.
from datetime import datetime  # noqa: TC003 — ORM mapper resolves at runtime
from typing import Any  # noqa: TC003 — ORM mapper resolves at runtime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Enum,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column  # noqa: TC002 — ORM mapper resolves at runtime

from database.base import Base

# The native enum message_role was DO-blocked by 0014; the Enum here only
# mirrors its name/values for autogenerate comparison — never re-created.
_MESSAGE_ROLE_VALUES = ("user", "assistant", "system")

_SESSION_STATUS_VALUES = (
    "active",
    "clarifying",
    "resumed",
    "completed",
    "abandoned_expired",
    "failed",
)

# Row kinds evolve, hence a CHECK (not a PG enum); the tuple is exported so the
# projector and its tests import one source of truth.
MESSAGE_KIND_VALUES = (
    "query",
    "clarification_prompt",
    "clarification_reply",
    "plan_prompt",
    "plan_decision",
    "final_summary",
    "turn_closed",
)
_KIND_CHECK_SQL = (
    "kind IN ('query', 'clarification_prompt', 'clarification_reply', "
    "'plan_prompt', 'plan_decision', 'final_summary', 'turn_closed')"
)

SESSION_STATUS_VALUES: tuple[str, ...] = _SESSION_STATUS_VALUES
MESSAGE_SCHEMA_VERSION = 1


class ChatSession(Base):
    """Store chat session metadata — one row per conversation (thread)."""

    __tablename__ = "chat_sessions"
    __table_args__: tuple[Any, ...] = (
        Index("ix_chat_sessions_user_updated", "user_id", text("updated_at DESC")),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    # 0022 dropped the redundant single-column index; the composite below leads
    # with user_id, so index=True must NOT come back on this column.
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # -- 0022: transcript projection cache (doc §3.1) -------------------------
    # status caches the latest turn's derived status; the truth is checkpoint
    # state read through the projector (§4.1), never written by the loop.
    schema_version: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("1")
    )
    status: Mapped[str] = mapped_column(
        Enum(*_SESSION_STATUS_VALUES, name="chat_session_status"),
        nullable=False,
        server_default=text("'active'"),
    )
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    last_checkpoint_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_checkpoint_ts: Mapped[datetime | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ChatMessage(Base):
    """Store chat messages between user and LLM — a projection of the checkpoint."""

    __tablename__ = "chat_messages"
    __table_args__: tuple[Any, ...] = (
        CheckConstraint(_KIND_CHECK_SQL, name="ck_chat_messages_kind"),
        # Item-level idempotency: one row per projected item across replays (§4.3).
        Index("uq_chat_messages_session_source_key", "session_id", "source_key", unique=True),
        # Display order is (turn_seq, ordinal), NOT insert order — reconciliation
        # inserts late and ordering must survive that (D3).
        Index("ix_chat_messages_session_turn", "session_id", "turn_seq", "ordinal"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # Native PG enum message_role — created by 0014 DO block; keep name stable.
    role: Mapped[str] = mapped_column(
        Enum(*_MESSAGE_ROLE_VALUES, name="message_role"), nullable=False
    )
    # Nullable since 0022: erasure stores NULL ("no content"), never "".
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # -- 0022: projection lineage (doc §3.2) ----------------------------------
    turn_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source_key: Mapped[str] = mapped_column(Text, nullable=False)
    source_checkpoint_id: Mapped[str] = mapped_column(Text, nullable=False)
    redacted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
