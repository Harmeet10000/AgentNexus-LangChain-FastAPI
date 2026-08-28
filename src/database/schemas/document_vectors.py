"""Document vectors — legacy, superseded by UnifiedDocument/Chunks.

Kept only so Base.metadata still declares `document_vectors` and
`alembic check` does not propose DROP. New code must use
`app.features.documents.model.UnifiedDocument/UnifiedChunk`.
Do not add new columns here.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class DocumentVector(Base):
    """Legacy store for Pinecone-era embeddings — deprecated."""

    __tablename__ = "document_vectors"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    vector_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    doc_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, name="metadata", nullable=True
    )
    # Legacy — keep Python-side handling to match 0014 (no server_default there).
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(nullable=False)
