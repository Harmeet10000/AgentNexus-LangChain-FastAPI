"""Realign chunk-kind partial indexes with persisted ingestion values.

Revision ID: 0021
Revises: 0020
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0021"
down_revision: str | tuple[str, ...] | None = "0020"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

_LEGACY_INDEXES = (
    "ix_chunks_kind_contracts",
    "ix_chunks_kind_statutes",
    "ix_chunks_kind_judgments",
    "ix_chunks_kind_filings",
)


def upgrade() -> None:
    for index_name in _LEGACY_INDEXES:
        op.drop_index(index_name, table_name="chunks")
    op.create_index(
        "ix_chunks_kind_legal_contract",
        "chunks",
        ["user_id"],
        postgresql_where=sa.text("chunk_kind = 'legal_contract'"),
    )
    op.create_index(
        "ix_chunks_kind_legal_policy",
        "chunks",
        ["user_id"],
        postgresql_where=sa.text("chunk_kind = 'legal_policy'"),
    )
    op.create_index(
        "ix_chunks_kind_generic",
        "chunks",
        ["user_id"],
        postgresql_where=sa.text("chunk_kind = 'generic'"),
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_kind_generic", table_name="chunks")
    op.drop_index("ix_chunks_kind_legal_policy", table_name="chunks")
    op.drop_index("ix_chunks_kind_legal_contract", table_name="chunks")
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
