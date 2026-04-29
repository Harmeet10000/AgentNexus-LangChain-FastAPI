"""
SQLAlchemy async models for the legal memory schema.

Tables:
  entities         → canonical legal entities (parties, clauses, contracts, obligations)
  relationships    → typed directed edges between entities (temporal, confidence-scored)
  parent_documents → one uploaded legal document per row
  clauses          → child chunks + pgvector embedding (768-dim) + pg_textsearch BM25 text
  events           → append-only audit log (immutable — dual-write partner to mutable state)
  memory_versions  → CRDT-lite versioning; written by ReconciliationAgent before/after snapshots

Alembic prerequisite (run once on DB):
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
Then: uv run alembic revision --autogenerate -m "add_memory_schema"
      uv run alembic upgrade head
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship  # noqa: TC002

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any


class Base(DeclarativeBase):
    pass


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("normalized_name", "entity_type", name="uq_entity_name_type"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_entity_confidence"),
        CheckConstraint("decay_score >= 0.0 AND decay_score <= 1.0", name="ck_entity_decay"),
        Index("idx_entities_normalized_name", "normalized_name"),
        Index("idx_entities_type", "entity_type"),
        Index("idx_entities_decay_score", "decay_score"),
        Index("idx_entities_doc_id", "doc_id"),
        Index("idx_entities_user_id", "user_id"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    decay_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    graphiti_episode_uuid: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    outgoing_relationships: Mapped[list[Relationship]] = relationship(
        "Relationship",
        foreign_keys="[Relationship.from_entity_id]",
        back_populates="from_entity",
        lazy="noload",
    )
    incoming_relationships: Mapped[list[Relationship]] = relationship(
        "Relationship",
        foreign_keys="[Relationship.to_entity_id]",
        back_populates="to_entity",
        lazy="noload",
    )
    versions: Mapped[list[MemoryVersion]] = relationship(
        "MemoryVersion", back_populates="entity", lazy="noload"
    )


class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_rel_confidence"),
        Index("idx_relationships_from_entity", "from_entity_id"),
        Index("idx_relationships_to_entity", "to_entity_id"),
        Index("idx_relationships_type", "relation_type"),
        Index("idx_relationships_doc_id", "doc_id"),
        Index("idx_relationships_valid_range", "valid_from", "valid_to"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    from_entity_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    to_entity_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    clause_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="graphiti_extraction")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    from_entity: Mapped[Entity] = relationship(
        "Entity", foreign_keys=[from_entity_id], back_populates="outgoing_relationships"
    )
    to_entity: Mapped[Entity] = relationship(
        "Entity", foreign_keys=[to_entity_id], back_populates="incoming_relationships"
    )


class ParentDocument(Base):
    """One uploaded file. Child retrieval chunks live in clauses."""

    __tablename__ = "parent_documents"
    __table_args__ = (
        UniqueConstraint("doc_id", name="uq_parent_documents_doc_id"),
        Index("idx_parent_documents_doc_id", "doc_id"),
        Index("idx_parent_documents_user_id", "user_id"),
        Index("idx_parent_documents_metadata_gin", "metadata_", postgresql_using="gin"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    document_type: Mapped[str] = mapped_column(String(128), nullable=False, default="unknown")
    jurisdiction: Mapped[str] = mapped_column(String(128), nullable=False, default="India")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    markdown: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    clauses: Mapped[list[Clause]] = relationship(
        "Clause", back_populates="parent_document", lazy="noload"
    )


class Clause(Base):
    """Child clause chunk with 768-dim embedding and pg_textsearch BM25 search text."""

    __tablename__ = "clauses"
    __table_args__ = (
        Index("idx_clauses_doc_id", "doc_id"),
        Index("idx_clauses_parent_doc_id", "parent_doc_id"),
        Index("idx_clauses_parent_chunk_index", "parent_doc_id", "chunk_index", unique=True),
        Index("idx_clauses_type", "clause_type"),
        Index("idx_clauses_risk_score", "risk_score"),
        Index("idx_clauses_user_id", "user_id"),
        Index("idx_clauses_metadata_gin", "metadata_", postgresql_using="gin"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    chunk_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), nullable=False, default=lambda: str(uuid4()), unique=True
    )
    parent_doc_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("parent_documents.id", ondelete="CASCADE"), nullable=True
    )
    contract_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    clause_id: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    preamble: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding: Mapped[Any] = mapped_column(Vector(768), nullable=True)
    clause_type: Mapped[str] = mapped_column(String(64), nullable=False, default="other")
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    custom_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    search_text: Mapped[str] = mapped_column(
        Text,
        Computed(
            "COALESCE(clause_type, '') || ' ' || "
            "COALESCE(preamble, '') || ' ' || "
            "COALESCE(chunk_text, text, '')",
            persisted=True,
        ),
        nullable=True,
    )
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decay_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    parent_document: Mapped[ParentDocument | None] = relationship(
        "ParentDocument", back_populates="clauses"
    )


class Event(Base):
    """Immutable append-only audit log. NEVER UPDATE or DELETE rows here."""

    __tablename__ = "events"
    __table_args__ = (
        Index("idx_events_doc_id", "doc_id"),
        Index("idx_events_user_id", "user_id"),
        Index("idx_events_event_type", "event_type"),
        Index("idx_events_thread_id", "thread_id"),
        Index("idx_events_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    doc_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MemoryVersion(Base):
    """Before/after snapshots written by ReconciliationAgent on every entity mutation.

    version is monotonically increasing per entity_id.
    To time-travel: SELECT * FROM memory_versions WHERE entity_id=X ORDER BY version ASC.
    """

    __tablename__ = "memory_versions"
    __table_args__ = (
        UniqueConstraint("entity_id", "version", name="uq_memory_version_entity_version"),
        Index("idx_memory_versions_entity_id", "entity_id"),
        Index("idx_memory_versions_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    entity_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    change_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # merge | update | correction
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    entity: Mapped[Entity] = relationship("Entity", back_populates="versions")
