"""add unified documents and chunks

Revision ID: a71f0d7d9c12
Revises: 2bc7726317f6
Create Date: 2026-06-13 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision = "a71f0d7d9c12"
down_revision = "2bc7726317f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=True),
        sa.Column("object_uri", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("document_kind", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("jurisdiction", sa.String(length=255), nullable=True),
        sa.Column("contract_type", sa.String(length=255), nullable=True),
        sa.Column("parties", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metadata_", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("user_id", "content_hash", name="uq_documents_user_content_hash"),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"], unique=False)
    op.create_index("ix_documents_kind", "documents", ["document_kind"], unique=False)
    op.create_index("ix_documents_status", "documents", ["status"], unique=False)
    op.execute("CREATE INDEX ix_documents_metadata_gin ON documents USING gin (metadata_)")

    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_kind", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("preamble", sa.Text(), nullable=False),
        sa.Column("clause_type", sa.String(length=128), nullable=True),
        sa.Column("page_no", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(dim=768), nullable=True),
        sa.Column("metadata_", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "custom_metadata", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "quality_warnings", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("graphiti_episode_id", sa.String(length=255), nullable=True),
        sa.Column("graphiti_verified", sa.Boolean(), nullable=False),
        sa.Column(
            "search_text",
            sa.Text(),
            sa.Computed(
                "COALESCE(clause_type, '') || ' ' || COALESCE(preamble, '') || ' ' || COALESCE(content, '')",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
            name=op.f("fk_chunks_document_id_documents"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
    )
    op.create_index("ix_chunks_user_document", "chunks", ["user_id", "document_id"], unique=False)
    op.create_index("ix_chunks_kind", "chunks", ["chunk_kind"], unique=False)
    op.create_index("ix_chunks_graphiti_verified", "chunks", ["graphiti_verified"], unique=False)
    op.execute("CREATE INDEX ix_chunks_metadata_gin ON chunks USING gin (metadata_)")
    op.execute(
        "CREATE INDEX chunks_bm25_idx ON chunks USING bm25(search_text) WITH (text_config='english', k1=1.2, b=0.75)"
    )
    op.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks USING diskann (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX chunks_search_text_trgm_idx ON chunks USING gin(search_text gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_search_text_trgm_idx")
    op.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    op.execute("DROP INDEX IF EXISTS chunks_bm25_idx")
    op.execute("DROP INDEX IF EXISTS ix_chunks_metadata_gin")
    op.drop_index("ix_chunks_graphiti_verified", table_name="chunks")
    op.drop_index("ix_chunks_kind", table_name="chunks")
    op.drop_index("ix_chunks_user_document", table_name="chunks")
    op.drop_table("chunks")

    op.execute("DROP INDEX IF EXISTS ix_documents_metadata_gin")
    op.drop_index("ix_documents_status", table_name="documents")
    op.drop_index("ix_documents_kind", table_name="documents")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_table("documents")
