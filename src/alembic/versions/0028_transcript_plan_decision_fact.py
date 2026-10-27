"""Persist validated, content-free plan decision evidence.

The pending fact is nullable so existing transcript rows remain readable and
legacy pauses fail closed (no fabricated decision). It is keyed to the
checkpoint's prompt source key and cleared with the decision-row transaction.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0028"
down_revision: str | tuple[str, ...] | None = "0027"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column("chat_sessions", sa.Column("pending_plan_decision_key", sa.Text(), nullable=True))
    op.add_column(
        "chat_sessions",
        sa.Column("pending_plan_decision_modified", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "pending_plan_decision_modified")
    op.drop_column("chat_sessions", "pending_plan_decision_key")
