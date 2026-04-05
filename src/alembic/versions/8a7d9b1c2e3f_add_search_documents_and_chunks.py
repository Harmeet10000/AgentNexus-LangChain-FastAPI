"""Add search documents and chunks schema

Revision ID: 8a7d9b1c2e3f
Revises: 2bc7726317f6
Create Date: 2026-04-05 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8a7d9b1c2e3f"
down_revision: str | None = "2bc7726317f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    op.create_table(
        "search_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("doc_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_search_documents")),
        sa.UniqueConstraint("content_hash", name=op.f("uq_search_documents_content_hash")),
    )

    op.create_table(
        "search_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("chunk_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["search_documents.id"],
            name=op.f("fk_search_chunks_document_id_search_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_search_chunks")),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_search_chunks_document_chunk_index",
        ),
    )

    op.create_index("ix_search_chunks_document_id", "search_chunks", ["document_id"], unique=False)
    op.create_index(
        "ix_search_chunks_document_chunk_index",
        "search_chunks",
        ["document_id", "chunk_index"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX search_chunks_embedding_idx ON search_chunks "
        "USING diskann (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX search_chunks_bm25_idx ON search_chunks "
        "USING bm25(content) WITH (text_config='english')"
    )
    op.execute(
        "CREATE INDEX ix_search_chunks_chunk_metadata_gin ON search_chunks "
        "USING gin (chunk_metadata)"
    )
    op.execute(
        "CREATE INDEX ix_search_chunks_content_tsv_gin ON search_chunks USING gin (content_tsv)"
    )
    op.execute(
        "CREATE INDEX ix_search_chunks_content_trgm ON search_chunks "
        "USING gin (content gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_search_chunks_content_trgm")
    op.execute("DROP INDEX IF EXISTS ix_search_chunks_content_tsv_gin")
    op.execute("DROP INDEX IF EXISTS ix_search_chunks_chunk_metadata_gin")
    op.execute("DROP INDEX IF EXISTS search_chunks_bm25_idx")
    op.execute("DROP INDEX IF EXISTS search_chunks_embedding_idx")
    op.drop_index("ix_search_chunks_document_chunk_index", table_name="search_chunks")
    op.drop_index("ix_search_chunks_document_id", table_name="search_chunks")
    op.drop_table("search_chunks")
    op.drop_table("search_documents")
