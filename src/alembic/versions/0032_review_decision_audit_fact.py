"""Allow append-only human review decision facts in transcripts."""

from __future__ import annotations

from alembic import op

revision: str = "0032"
down_revision: str | tuple[str, ...] | None = "0031"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # The naming convention changes an explicit check name when using
    # ``drop_constraint``. Revision 0014 created the catalog name directly.
    op.execute("ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS ck_chat_messages_kind")
    op.execute(
        "ALTER TABLE chat_messages ADD CONSTRAINT ck_chat_messages_kind CHECK "
        "(kind IN ('query', 'clarification_prompt', 'clarification_reply', "
        "'plan_prompt', 'plan_decision', 'review_decision', 'final_summary', 'turn_closed'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS ck_chat_messages_kind")
    op.execute(
        "ALTER TABLE chat_messages ADD CONSTRAINT ck_chat_messages_kind CHECK "
        "(kind IN ('query', 'clarification_prompt', 'clarification_reply', "
        "'plan_prompt', 'plan_decision', 'final_summary', 'turn_closed'))"
    )
