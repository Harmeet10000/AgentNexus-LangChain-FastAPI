"""SQLAlchemy models for site-search documents and chunks.

Ported from the legacy ``logs/search/model.py`` with two changes the legacy
feature lacked: every row carries ``tenant_id`` (all reads and writes are
tenant-scoped), and ``search_documents`` carries a ``status`` column so ingest
progress survives without trusting Celery result metadata.

The three retrieval indexes (bm25 / diskann / trgm) and the GIN indexes are
declared here, not only in the migration, for the same reason ``chunks``
declares them (see ``features/documents/domain/chunk.py``): an index present in
the database and absent from the registry is one ``alembic check`` proposes to
REMOVE — and ``search_chunks_bm25_idx`` is passed to ``to_bm25query()`` as a
literal string, so a silent drop would make keyword retrieval match nothing
without raising.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4  # noqa: TC003 — UUID resolved at runtime by SQLAlchemy mapper

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
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
from sqlalchemy.orm import (  # noqa: TC002 — Mapped resolved at runtime by SQLAlchemy mapper
    Mapped,
    mapped_column,
    relationship,
)

from database.base import Base

#: The width ``search_chunks.embedding`` is created with. The legacy feature
#: pinned Gemini output at 768 dims; the column type is concrete for the life of
#: the process, so it stays a literal rather than a settings read.
SEARCH_EMBEDDING_DIM = 768


class SearchDocument(Base):
    """Parent document record used for deduplication and context assembly."""

    __tablename__ = "search_documents"
    __table_args__ = (
        # Dedupe is per tenant: the same content may be ingested by two tenants,
        # and one tenant must never learn of another's document through a
        # uniqueness violation.
        UniqueConstraint(
            "tenant_id",
            "content_hash",
            name="uq_search_documents_tenant_content_hash",
        ),
        Index("ix_search_documents_tenant_id", "tenant_id"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(length=255), nullable=False, default="")
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str] = mapped_column(String(length=500), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(length=64), nullable=False)
    status: Mapped[str] = mapped_column(String(length=32), nullable=False, default="pending")
    doc_metadata: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
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

    chunks: Mapped[list[SearchChunk]] = relationship(
        "SearchChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SearchChunk(Base):
    """Searchable child chunk record."""

    __tablename__ = "search_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_search_chunks_document_chunk_index",
        ),
        Index("ix_search_chunks_tenant_id", "tenant_id"),
        Index("ix_search_chunks_document_id", "document_id"),
        Index("ix_search_chunks_chunk_metadata_gin", "chunk_metadata", postgresql_using="gin"),
        Index(
            "ix_search_chunks_content_trgm",
            "content",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
        ),
        Index(
            "search_chunks_bm25_idx",
            "content",
            postgresql_using="bm25",
            # Values render through str() unquoted, so 'english' carries its own
            # quotes — a bare "english" would emit an unquoted identifier.
            postgresql_with={"text_config": "'english'"},
        ),
        Index(
            "search_chunks_embedding_idx",
            "embedding",
            postgresql_using="diskann",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(length=255), nullable=False, default="")
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("search_documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(SEARCH_EMBEDDING_DIM), nullable=True
    )
    chunk_metadata: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    # No text-search vector column: the lexical leg is pg_textsearch's bm25 access method
    # directly on ``content`` (see search_repository.bm25_search). A full-text vector
    # would double-count the lexical signal in the three-branch fusion — the
    # same reason the no-full-text-vector gate bans it under src/app.
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

    document: Mapped[SearchDocument] = relationship("SearchDocument", back_populates="chunks")
