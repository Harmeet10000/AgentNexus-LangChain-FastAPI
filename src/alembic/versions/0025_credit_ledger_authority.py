"""Make downgrade notes authoritative and applications idempotent.

Revision ID: 0025
Revises: 0024

This migration is intentionally offline-only: it describes schema changes but
does not connect to or execute against a live database.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0025"
down_revision: str | tuple[str, ...] | None = "0024"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # One active row per user made multiple independent grants impossible.
    op.drop_index("uq_user_credits_user_id_active", table_name="user_credits")
    op.drop_constraint(
        "ck_user_credits_remaining_balance_lte_amount", "user_credits", type_="check"
    )
    op.create_check_constraint(
        "ck_user_credits_remaining_balance_valid",
        "user_credits",
        "remaining_balance >= 0 AND remaining_balance <= credit_amount",
    )
    op.create_check_constraint(
        "ck_user_credits_status",
        "user_credits",
        "status IN ('active', 'consumed', 'expired')",
    )
    op.create_check_constraint(
        "ck_user_credits_validity_range",
        "user_credits",
        "valid_until IS NULL OR valid_from <= valid_until",
    )

    op.add_column(
        "credit_notes",
        sa.Column("credit_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_credit_notes_credit_id",
        "credit_notes",
        "user_credits",
        ["credit_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index("ix_credit_notes_credit_id", "credit_notes", ["credit_id"])

    op.create_table(
        "credit_note_applications",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("credit_note_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("invoice_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("amount_paisa", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["credit_note_id"], ["credit_notes.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["invoice_id"], ["invoices.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "credit_note_id",
            "invoice_id",
            name="uq_credit_note_applications_note_invoice",
        ),
        sa.CheckConstraint("amount_paisa > 0", name="ck_credit_note_applications_amount_positive"),
    )
    op.create_index(
        "ix_credit_note_applications_invoice_id",
        "credit_note_applications",
        ["invoice_id"],
    )
    op.create_index(
        "ix_credit_note_applications_credit_note_id",
        "credit_note_applications",
        ["credit_note_id"],
    )

    op.add_column(
        "invoices",
        sa.Column(
            "credit_applied",
            sa.Numeric(20, 2),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("invoices", "credit_applied")
    op.drop_index(
        "ix_credit_note_applications_credit_note_id", table_name="credit_note_applications"
    )
    op.drop_index("ix_credit_note_applications_invoice_id", table_name="credit_note_applications")
    op.drop_table("credit_note_applications")
    op.drop_index("ix_credit_notes_credit_id", table_name="credit_notes")
    op.drop_constraint("fk_credit_notes_credit_id", "credit_notes", type_="foreignkey")
    op.drop_column("credit_notes", "credit_id")
    op.drop_constraint("ck_user_credits_validity_range", "user_credits", type_="check")
    op.drop_constraint("ck_user_credits_status", "user_credits", type_="check")
    op.drop_constraint("ck_user_credits_remaining_balance_valid", "user_credits", type_="check")
    op.create_check_constraint(
        "ck_user_credits_remaining_balance_lte_amount",
        "user_credits",
        "remaining_balance <= credit_amount",
    )
    op.create_index(
        "uq_user_credits_user_id_active",
        "user_credits",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active' AND deleted_at IS NULL"),
    )
