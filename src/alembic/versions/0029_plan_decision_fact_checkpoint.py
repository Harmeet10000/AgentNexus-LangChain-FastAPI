"""Bind pending plan-decision evidence to its checkpoint generation."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0029"
down_revision: str | tuple[str, ...] | None = "0028"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("pending_plan_decision_checkpoint_id", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "pending_plan_decision_checkpoint_id")
