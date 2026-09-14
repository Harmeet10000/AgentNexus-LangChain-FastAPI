"""Persist document structure and extraction completion state.

Revision ID: 0020
Revises: 0019
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0020"
down_revision: str | tuple[str, ...] | None = "0019"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "structural_tree",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.alter_column("documents", "structural_tree", server_default=None)
    op.add_column(
        "documents",
        sa.Column(
            "extraction_incomplete",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.alter_column("documents", "extraction_incomplete", server_default=None)


def downgrade() -> None:
    op.drop_column("documents", "extraction_incomplete")
    op.drop_column("documents", "structural_tree")
