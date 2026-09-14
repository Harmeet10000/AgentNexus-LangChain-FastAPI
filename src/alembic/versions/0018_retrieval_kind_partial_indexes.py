"""Partial tenant-first indexes for the stable document-kind values.

Revision ID: 0018
Revises: 0017

First rung of the retrieval-sql isolation ladder (ADR-004): partial indexes for
the stable, closed set of document-kind values the retrieval filters
constrain. Each index leads with ``user_id`` so tenant-scoped, kind-filtered
reads narrow to one tenant's rows of one kind; the partial predicate keeps
every other kind off the index entirely.

The extension block repeats ``CREATE EXTENSION IF NOT EXISTS`` for all four
retrieval extensions, following revision 0013's pattern: on a fresh database
built by ``alembic upgrade head`` with no pre-installed extensions, the chain
must reach head without assuming the hosting image. Every statement is
idempotent.

This change adds no columns (ADR-008).
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0018"
down_revision: str | tuple[str, ...] | None = "0017"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    for extension in ("vector", "vectorscale", "pg_trgm", "pg_textsearch"):
        op.execute(sa.text(f"CREATE EXTENSION IF NOT EXISTS {extension}"))
    op.create_index(
        "ix_chunks_kind_contracts",
        "chunks",
        ["user_id"],
        postgresql_where=sa.text("chunk_kind = 'contracts'"),
    )
    op.create_index(
        "ix_chunks_kind_statutes",
        "chunks",
        ["user_id"],
        postgresql_where=sa.text("chunk_kind = 'statutes'"),
    )
    op.create_index(
        "ix_chunks_kind_judgments",
        "chunks",
        ["user_id"],
        postgresql_where=sa.text("chunk_kind = 'judgments'"),
    )
    op.create_index(
        "ix_chunks_kind_filings",
        "chunks",
        ["user_id"],
        postgresql_where=sa.text("chunk_kind = 'filings'"),
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_kind_filings", table_name="chunks")
    op.drop_index("ix_chunks_kind_judgments", table_name="chunks")
    op.drop_index("ix_chunks_kind_statutes", table_name="chunks")
    op.drop_index("ix_chunks_kind_contracts", table_name="chunks")
