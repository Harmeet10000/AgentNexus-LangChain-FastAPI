"""Persist canonical-pipeline and Graphiti indexing provenance on chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0031"
down_revision: str | Sequence[str] | None = "0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "pipeline_version",
            sa.String(length=128),
            nullable=False,
            server_default="documents-v1",
        ),
    )
    op.add_column(
        "chunks",
        sa.Column("graph_indexed_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Existing verified rows predate the timestamp.  ``updated_at`` is the
    # closest durable observation available during migration; unverified rows
    # stay NULL and retain the legacy boolean as their compatibility signal.
    op.execute(
        """
        UPDATE chunks
        SET graph_indexed_at = updated_at
        WHERE graphiti_verified IS TRUE AND graph_indexed_at IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("chunks", "graph_indexed_at")
    op.drop_column("chunks", "pipeline_version")
