"""Add credit tables.

Revision ID: 0005
Revises: 0009
Create Date: 2026-08-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def _jsonb() -> postgresql.JSONB:
    return postgresql.JSONB(astext_type=sa.Text())


def _now() -> sa.DateTime:
    return sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "user_credits",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("credit_type", sa.String(length=32), nullable=False),
        sa.Column("credit_amount", sa.BigInteger(), nullable=False),
        sa.Column("remaining_balance", sa.BigInteger(), nullable=False),
        sa.Column("valid_from", _now(), nullable=False),
        sa.Column("valid_until", _now(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("consumed_at", _now(), nullable=True),
        sa.Column("metadata_", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("deleted_at", _now(), nullable=True),
        sa.Column("created_at", _now(), nullable=False),
        sa.Column("updated_at", _now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("credit_amount > 0", name="ck_user_credits_credit_amount_positive"),
        sa.CheckConstraint(
            "remaining_balance <= credit_amount",
            name="ck_user_credits_remaining_balance_lte_amount",
        ),
    )
    op.create_index("ix_user_credits_user_id", "user_credits", ["user_id"])
    op.create_index("ix_user_credits_status", "user_credits", ["status"])
    op.create_index("ix_user_credits_valid_until", "user_credits", ["valid_until"])
    op.create_index("ix_user_credits_created_at", "user_credits", ["created_at"])
    op.execute(
        "CREATE UNIQUE INDEX uq_user_credits_user_id_active ON user_credits (user_id) "
        "WHERE status = 'active' AND deleted_at IS NULL"
    )

    op.create_table(
        "credit_consumptions",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("credit_id", _uuid(), nullable=False),
        sa.Column("invoice_id", _uuid(), nullable=True),
        sa.Column("razorpay_payment_id", sa.String(length=64), nullable=True),
        sa.Column("consumed_amount", sa.BigInteger(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("metadata_", _jsonb(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", _now(), nullable=False),
        sa.ForeignKeyConstraint(["credit_id"], ["user_credits.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["invoice_id"], ["invoices.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "consumed_amount > 0", name="ck_credit_consumptions_consumed_amount_positive"
        ),
    )
    op.create_index("ix_credit_consumptions_user_id", "credit_consumptions", ["user_id"])
    op.create_index("ix_credit_consumptions_credit_id", "credit_consumptions", ["credit_id"])
    op.create_index("ix_credit_consumptions_invoice_id", "credit_consumptions", ["invoice_id"])
    op.create_index("ix_credit_consumptions_created_at", "credit_consumptions", ["created_at"])


def downgrade() -> None:
    op.drop_table("credit_consumptions")
    op.drop_table("user_credits")
