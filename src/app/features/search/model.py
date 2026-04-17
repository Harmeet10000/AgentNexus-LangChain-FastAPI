"""SQLAlchemy models for search documents and chunks."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4  # noqa: TC003

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship  # noqa: TC002

from app.shared import Base


class SearchDocument(Base):
    """Parent document record used for deduplication and context assembly."""

    __tablename__ = "search_documents"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(String(length=500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(length=64), unique=True, nullable=False)
    doc_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
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

    chunks: Mapped[list[SearchChunk]] = relationship(back_populates="document")


class SearchChunk(Base):
    """Searchable child chunk record."""

    __tablename__ = "search_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_search_chunks_document_chunk_index"
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("search_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=False,
    )
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

    document: Mapped[SearchDocument] = relationship(back_populates="chunks")
