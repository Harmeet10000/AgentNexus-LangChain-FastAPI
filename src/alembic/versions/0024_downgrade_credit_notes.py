"""Persist idempotent downgrade credit notes.

Revision ID: 0024
Revises: 0023
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0024"
down_revision: str | tuple[str, ...] | None = "0023"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "credit_notes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("subscription_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_invoice_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("operation_key", sa.String(length=128), nullable=False),
        sa.Column("amount_paisa", sa.BigInteger(), nullable=False),
        sa.Column("remaining_paisa", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("applies_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="available"),
        sa.Column(
            "metadata_", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["source_invoice_id"], ["invoices.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["subscription_id"], ["subscriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("operation_key", name="uq_credit_notes_operation_key"),
        sa.CheckConstraint("amount_paisa > 0", name="ck_credit_notes_amount_positive"),
        sa.CheckConstraint(
            "remaining_paisa >= 0 AND remaining_paisa <= amount_paisa",
            name="ck_credit_notes_remaining_valid",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'available', 'applied', 'cancelled')",
            name="ck_credit_notes_status",
        ),
    )
    op.create_index("ix_credit_notes_subscription_id", "credit_notes", ["subscription_id"])
    op.create_index("ix_credit_notes_source_invoice_id", "credit_notes", ["source_invoice_id"])
    op.create_index("ix_credit_notes_applies_at", "credit_notes", ["applies_at"])


def downgrade() -> None:
    op.drop_index("ix_credit_notes_applies_at", table_name="credit_notes")
    op.drop_index("ix_credit_notes_source_invoice_id", table_name="credit_notes")
    op.drop_index("ix_credit_notes_subscription_id", table_name="credit_notes")
    op.drop_table("credit_notes")
