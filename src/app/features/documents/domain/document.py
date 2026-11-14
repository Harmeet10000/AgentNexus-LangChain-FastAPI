"""Unified document SQLAlchemy model — the retrieval-truth document aggregate."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4  # noqa: TC003 — UUID resolved at runtime by SQLAlchemy mapper

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
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

if TYPE_CHECKING:
    from typing import Final

    from app.features.documents.domain.chunk import UnifiedChunk


class DocumentStatus(StrEnum):
    """Typed queued-ingestion lifecycle status (harden 3.11).

    The ``documents.status`` column stores these exact string values; the
    repository refuses any other value so a document can never drift into an
    untyped status that downstream readiness gates cannot interpret.
    """

    RECEIVED = "received"
    PROCESSING = "processing"
    PARSED = "parsed"
    STORED_POSTGRES = "stored_postgres"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    # STRIDE D-3: the embedding provider's daily quota was exhausted mid-run.
    # Deliberately NON-terminal — no seconds-scale backoff clears a daily cap,
    # so the retry is the next quota window's idempotent resume, which selects
    # non-terminal rows. Read-side surfaces fall through to the retryable
    # ``index_not_ready`` state (``evidence/service._ingestion_state``). A
    # ``quota_exhausted`` row carries typed metadata (``ingestion_error`` with
    # code ``embedding_quota_exhausted``), never a bare status.
    QUOTA_EXHAUSTED = "quota_exhausted"


DOCUMENT_STATUS_VALUES: Final[frozenset[str]] = frozenset(status.value for status in DocumentStatus)

TERMINAL_DOCUMENT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        DocumentStatus.COMPLETED.value,
        DocumentStatus.COMPLETED_WITH_WARNINGS.value,
        DocumentStatus.FAILED.value,
    }
)

# Metadata keys that carry a typed ingestion/extraction failure reason. The
# repository's extraction-reason guard consults these before letting an
# ``extraction_incomplete=True`` write through without a reason.
FAILURE_REASON_METADATA_KEYS: Final[tuple[str, ...]] = (
    "knowledge_extraction_failure",
    "ingestion_error",
    "extraction_failure_reason",
)

# Persisted when a write marks extraction incomplete but carries no typed
# reason — the live-run diagnosability gap (12/15 corpus documents failed with
# ``extraction_incomplete: true`` and NO reason stored).
EXTRACTION_FAILURE_UNRECORDED: Final[str] = "EXTRACTION_FAILURE_UNRECORDED"


class UnifiedDocument(Base):
    """Single retrieval-truth parent row for uploaded or ingested documents."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "content_hash", name="uq_documents_tenant_content_hash"),
        # Retained for rolling deployments and audit queries; ownership is
        # enforced by tenant_id, not uploader identity.
        Index("ix_documents_user_id", "user_id"),
        Index("ix_documents_tenant_id", "tenant_id"),
        Index("ix_documents_kind", "document_kind"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_metadata_gin", "metadata_", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[str] = mapped_column(String(length=255), nullable=False, default="")
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
    structural_tree: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    extraction_incomplete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
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

    # The related class is named as a string, not imported: the two aggregates
    # live in sibling modules (``domain.chunk``) and a runtime import here would
    # be circular. SQLAlchemy resolves the name against the shared registry once
    # both modules are imported (see ``database.registry.MODEL_MODULE_NAMES``).
    chunks: Mapped[list[UnifiedChunk]] = relationship(
        "UnifiedChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
