"""Add document_version and locus to chunks; widen unique identity.

Revision ID: 0019
Revises: 0018

Chunk identity becomes ``(document_id, document_version, chunk_index)`` so two
versions of one document coexist with distinguishable rows. ``locus`` stores a
recovered structural address (clause numbering); unknown position is NULL, never
an empty string.

Existing rows receive ``document_version = 1``. The unique constraint is renamed
in place: drop ``uq_chunks_document_chunk_index``, add
``uq_chunks_document_version_chunk_index``.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0019"
down_revision: str | tuple[str, ...] | None = "0018"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("document_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "chunks",
        sa.Column("locus", sa.String(length=255), nullable=True),
    )
    op.drop_constraint("uq_chunks_document_chunk_index", "chunks", type_="unique")
    op.create_unique_constraint(
        "uq_chunks_document_version_chunk_index",
        "chunks",
        ["document_id", "document_version", "chunk_index"],
    )
    # server_default was only for the backfill of existing rows; new writes
    # supply the value explicitly from the ORM / write path.
    op.alter_column("chunks", "document_version", server_default=None)


def downgrade() -> None:
    op.drop_constraint("uq_chunks_document_version_chunk_index", "chunks", type_="unique")
    op.create_unique_constraint(
        "uq_chunks_document_chunk_index",
        "chunks",
        ["document_id", "chunk_index"],
    )
    op.drop_column("chunks", "locus")
    op.drop_column("chunks", "document_version")
