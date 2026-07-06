"""Unified document and chunk SQLAlchemy models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import mapped_column
from app.shared import Base

if TYPE_CHECKING:
    from sqlalchemy.orm import Mapped, relationship
    from uuid import UUID


class UnifiedDocument(Base):
    """Single retrieval-truth parent row for uploaded or ingested documents."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("user_id", "content_hash", name="uq_documents_user_content_hash"),
        Index("ix_documents_user_id", "user_id"),
        Index("ix_documents_kind", "document_kind"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_metadata_gin", "metadata_", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    title: Mapped[str] = mapped_column(String(length=500), nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    object_uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(length=64), nullable=False)
    document_kind: Mapped[str] = mapped_column(String(length=64), nullable=False, default="generic")
    status: Mapped[str] = mapped_column(String(length=64), nullable=False, default="received")
    jurisdiction: Mapped[str | None] = mapped_column(String(length=255), nullable=True)
    contract_type: Mapped[str | None] = mapped_column(String(length=255), nullable=True)
    parties: Mapped[list[object]] = mapped_column(JSONB, default=list, nullable=False)
    metadata_: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    chunks: Mapped[list[UnifiedChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class UnifiedChunk(Base):
    """Single retrieval-truth searchable chunk row."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
        Index("ix_chunks_user_document", "user_id", "document_id"),
        Index("ix_chunks_kind", "chunk_kind"),
        Index("ix_chunks_metadata_gin", "metadata_", postgresql_using="gin"),
        Index("ix_chunks_graphiti_verified", "graphiti_verified"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_kind: Mapped[str] = mapped_column(String(length=64), nullable=False, default="generic")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    preamble: Mapped[str] = mapped_column(Text, nullable=False, default="")
    clause_type: Mapped[str | None] = mapped_column(String(length=128), nullable=True)
    page_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    metadata_: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    custom_metadata: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    quality_warnings: Mapped[list[object]] = mapped_column(JSONB, default=list, nullable=False)
    graphiti_episode_id: Mapped[str | None] = mapped_column(String(length=255), nullable=True)
    graphiti_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    search_text: Mapped[str] = mapped_column(
        Text,
        Computed(
            "COALESCE(clause_type, '') || ' ' || "
            "COALESCE(preamble, '') || ' ' || "
            "COALESCE(content, '')",
            persisted=True,
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    document: Mapped[UnifiedDocument] = relationship(back_populates="chunks")
