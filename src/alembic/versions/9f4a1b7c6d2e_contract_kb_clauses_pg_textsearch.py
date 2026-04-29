"""Contract KB parent documents and pg_textsearch clauses.

Revision ID: 9f4a1b7c6d2e
Revises: 8a7d9b1c2e3f
Create Date: 2026-04-20 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "9f4a1b7c6d2e"
down_revision: str | None = "8a7d9b1c2e3f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")

    op.create_table(
        "parent_documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("document_type", sa.String(length=128), nullable=False),
        sa.Column("jurisdiction", sa.String(length=128), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_parent_documents")),
        sa.UniqueConstraint("doc_id", name=op.f("uq_parent_documents_doc_id")),
    )
    op.create_index("idx_parent_documents_doc_id", "parent_documents", ["doc_id"], unique=False)
    op.create_index("idx_parent_documents_user_id", "parent_documents", ["user_id"], unique=False)
    op.create_index(
        "idx_parent_documents_metadata_gin",
        "parent_documents",
        ["metadata"],
        unique=False,
        postgresql_using="gin",
    )

    with op.batch_alter_table("clauses") as batch_op:
        batch_op.add_column(sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=True))
        batch_op.add_column(
            sa.Column("parent_doc_id", postgresql.UUID(as_uuid=True), nullable=True)
        )
        batch_op.add_column(sa.Column("chunk_index", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("preamble", sa.Text(), nullable=False, server_default=""))
        batch_op.add_column(sa.Column("chunk_text", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "metadata_",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default="{}",
            )
        )
        batch_op.add_column(
            sa.Column(
                "custom_metadata",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default="{}",
            )
        )
        batch_op.add_column(
            sa.Column(
                "search_text",
                sa.Text(),
                sa.Computed(
                    "COALESCE(clause_type, '') || ' ' || "
                    "COALESCE(preamble, '') || ' ' || "
                    "COALESCE(chunk_text, text, '')",
                    persisted=True,
                ),
                nullable=True,
            )
        )

    op.execute("UPDATE clauses SET chunk_id = id WHERE chunk_id IS NULL")
    op.execute("UPDATE clauses SET chunk_text = text WHERE chunk_text IS NULL")
    op.alter_column("clauses", "chunk_id", nullable=False)
    op.alter_column("clauses", "chunk_text", nullable=False)
    op.alter_column("clauses", "embedding", type_=Vector(768), existing_nullable=True)

    op.create_foreign_key(
        "fk_clauses_parent_doc_id_parent_documents",
        "clauses",
        "parent_documents",
        ["parent_doc_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("idx_clauses_parent_doc_id", "clauses", ["parent_doc_id"], unique=False)
    op.create_index("idx_clauses_chunk_id", "clauses", ["chunk_id"], unique=True)
    op.create_index(
        "idx_clauses_parent_chunk_index",
        "clauses",
        ["parent_doc_id", "chunk_index"],
        unique=True,
    )
    op.create_index(
        "idx_clauses_metadata_gin",
        "clauses",
        ["metadata_"],
        unique=False,
        postgresql_using="gin",
    )
    op.execute("DROP INDEX IF EXISTS clauses_bm25_idx")
    op.execute(
        "CREATE INDEX clauses_bm25_idx ON clauses "
        "USING bm25(search_text) "
        "WITH (text_config = 'english', k1 = 1.2, b = 0.75)"
    )
    op.execute("DROP INDEX IF EXISTS clauses_embedding_idx")
    op.execute(
        "CREATE INDEX clauses_embedding_idx ON clauses USING diskann (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS clauses_embedding_idx")
    op.execute("DROP INDEX IF EXISTS clauses_bm25_idx")
    op.drop_index("idx_clauses_metadata_gin", table_name="clauses")
    op.drop_index("idx_clauses_chunk_id", table_name="clauses")
    op.drop_index("idx_clauses_parent_chunk_index", table_name="clauses")
    op.drop_index("idx_clauses_parent_doc_id", table_name="clauses")
    op.drop_constraint("fk_clauses_parent_doc_id_parent_documents", "clauses", type_="foreignkey")
    with op.batch_alter_table("clauses") as batch_op:
        batch_op.drop_column("search_text")
        batch_op.drop_column("custom_metadata")
        batch_op.drop_column("metadata_")
        batch_op.drop_column("chunk_text")
        batch_op.drop_column("preamble")
        batch_op.drop_column("chunk_index")
        batch_op.drop_column("parent_doc_id")
        batch_op.drop_column("chunk_id")
    op.drop_index("idx_parent_documents_metadata_gin", table_name="parent_documents")
    op.drop_index("idx_parent_documents_user_id", table_name="parent_documents")
    op.drop_index("idx_parent_documents_doc_id", table_name="parent_documents")
    op.drop_table("parent_documents")
