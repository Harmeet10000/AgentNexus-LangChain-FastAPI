"""Unified document and chunk SQLAlchemy models."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4  # noqa: TC003 — UUID resolved at runtime by SQLAlchemy mapper

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
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import (  # noqa: TC002 — Mapped resolved at runtime by SQLAlchemy mapper
    Mapped,
    mapped_column,
    relationship,
)

from app.config import get_settings
from database.base import Base

# The width `chunks.embedding` is created with, captured once when this module is
# imported. Two reasons it is a named constant rather than an inline call:
#
# 1. It is the *stored* width. `get_settings()` is re-read on every call, but a
#    SQLAlchemy column type is concrete for the life of the process — so the
#    moment configuration changes underneath a running app, this constant and
#    `EMBEDDING_DIMENSION` disagree, and that disagreement is exactly what the
#    write guard in `repository._reject_width_mismatch` detects.
# 2. Reading it back off the column (`__table__.c.embedding.type.dim`) is typed
#    as the base `TypeEngine`, which declares no `dim` — so every consumer would
#    need a narrowing dance or a suppression to get an `int` out.
#
# Not annotated `Final`: `from __future__ import annotations` makes the annotation
# a string that is never evaluated, so the import would be typing-only and `TC003`
# would ask for a type-checking block. The module-level SCREAMING_SNAKE name is
# how this codebase already spells a constant.
CHUNK_EMBEDDING_DIM = get_settings().EMBEDDING_DIMENSION


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
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        onupdate=lambda: datetime.now(tz=UTC),
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
        # The three retrieval branches. Declared here, not only in the migration,
        # because an index present in the database and absent from the registry is
        # one `alembic check` proposes to REMOVE — and `chunks_bm25_idx` is passed
        # to `to_bm25query()` as a literal string, so a silent drop would make
        # keyword retrieval match nothing without raising. The names are a query
        # contract; see `features/documents/repository.py`.
        # bm25 comes from pg_textsearch, diskann from vectorscale, gin_trgm_ops
        # from pg_trgm — all three created by revision a5bd6b69a28e.
        Index(
            "chunks_bm25_idx",
            "search_text",
            postgresql_using="bm25",
            # Values render through str() unquoted, so 'english' carries its own
            # quotes — a bare "english" would emit an unquoted identifier.
            postgresql_with={"text_config": "'english'", "k1": "1.2", "b": "0.75"},
        ),
        Index(
            "chunks_embedding_idx",
            "embedding",
            postgresql_using="diskann",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "chunks_search_text_trgm_idx",
            "search_text",
            postgresql_using="gin",
            postgresql_ops={"search_text": "gin_trgm_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey(column="documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(String(length=255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_kind: Mapped[str] = mapped_column(String(length=64), nullable=False, default="generic")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    preamble: Mapped[str] = mapped_column(Text, nullable=False, default="")
    clause_type: Mapped[str | None] = mapped_column(String(length=128), nullable=True)
    page_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Width comes from the one configured value, not a literal. It resolves to the
    # same 768 today, so this renders identically and `alembic check` proposes
    # nothing — which is the point: this is the cheapest possible moment to remove
    # the literal, before any row exists to migrate. Read at class-definition time
    # because a SQLAlchemy column type must be concrete when the class body runs.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(CHUNK_EMBEDDING_DIM), nullable=True
    )
    metadata_: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    custom_metadata: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    quality_warnings: Mapped[list[object]] = mapped_column(JSONB, default=list, nullable=False)
    graphiti_episode_id: Mapped[str | None] = mapped_column(String(length=255), nullable=True)
    graphiti_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    search_text: Mapped[str] = mapped_column(
        Text,
        Computed(
            sqltext="COALESCE(clause_type, '') || ' ' || "
            "COALESCE(preamble, '') || ' ' || "
            "COALESCE(content, '')",
            persisted=True,
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        nullable=False,
    )
    # Server-side default, unlike UnifiedDocument's Python-side one: revision
    # a5bd6b69a28e added this column as NOT NULL DEFAULT now() precisely so the
    # database can satisfy it alone. Declaring it here closes the inverse
    # hazard — a column in the database and absent from the registry is one
    # `alembic check` proposes to DROP, which loses data rather than speed.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    document: Mapped[UnifiedDocument] = relationship(back_populates="chunks")
