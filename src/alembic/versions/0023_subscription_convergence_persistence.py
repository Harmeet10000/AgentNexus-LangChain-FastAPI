"""Persist subscription convergence state and provider operation outcomes.

Revision ID: 0023
Revises: 0022

The existing ``subscriptions.plan_id`` remains the desired local plan for
backward compatibility.  ``observed_plan_id`` records the provider's last
normalized observation; it is deliberately not a foreign key because a
provider can report a plan that has not (yet) been imported locally.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0023"
down_revision: str | tuple[str, ...] | None = "0022"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def _uuid() -> postgresql.UUID:
    return postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    op.add_column(
        "subscriptions",
        sa.Column("observed_plan_id", _uuid(), nullable=True),
    )
    op.add_column(
        "subscriptions",
        sa.Column(
            "remote_sync",
            sa.String(length=32),
            nullable=True,
        ),
    )
    # Existing rows have no provider convergence proof. They must be
    # reconciled before being treated as synced; new rows also begin pending
    # until an authoritative provider observation proves convergence.
    op.execute("UPDATE subscriptions SET remote_sync = 'pending' WHERE remote_sync IS NULL")
    op.alter_column(
        "subscriptions",
        "remote_sync",
        nullable=False,
        server_default=sa.text("'pending'"),
    )
    op.add_column(
        "subscriptions",
        sa.Column("cancellation_requested_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "subscriptions",
        sa.Column("entitlement_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_subscriptions_observed_plan_id", "subscriptions", ["observed_plan_id"])
    op.create_index("ix_subscriptions_remote_sync", "subscriptions", ["remote_sync"])
    op.create_check_constraint(
        "ck_subscriptions_remote_sync",
        "subscriptions",
        "remote_sync IN ('synced', 'pending', 'retryable_failure', 'permanent_failure')",
    )

    op.create_table(
        "provider_operation_records",
        sa.Column("id", _uuid(), nullable=False),
        sa.Column("subscription_id", _uuid(), nullable=True),
        sa.Column("correlation_id", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("operation_kind", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("provider_resource_id", sa.String(length=128), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=1000), nullable=True),
        sa.Column(
            "request_data",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "observation_data",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["subscription_id"], ["subscriptions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "idempotency_key", name="uq_provider_operation_records_idempotency_key"
        ),
        sa.CheckConstraint(
            "state IN ('pending', 'succeeded', 'unknown', 'reconciliation', 'manual_review', 'failed')",
            name="ck_provider_operation_records_state",
        ),
    )
    op.create_index(
        "ix_provider_operation_records_subscription_id",
        "provider_operation_records",
        ["subscription_id"],
    )
    op.create_index(
        "ix_provider_operation_records_correlation_id",
        "provider_operation_records",
        ["correlation_id"],
    )
    op.create_index("ix_provider_operation_records_state", "provider_operation_records", ["state"])


def downgrade() -> None:
    op.drop_index("ix_provider_operation_records_state", table_name="provider_operation_records")
    op.drop_index(
        "ix_provider_operation_records_correlation_id", table_name="provider_operation_records"
    )
    op.drop_index(
        "ix_provider_operation_records_subscription_id", table_name="provider_operation_records"
    )
    op.drop_table("provider_operation_records")
    op.drop_constraint("ck_subscriptions_remote_sync", "subscriptions", type_="check")
    op.drop_index("ix_subscriptions_remote_sync", table_name="subscriptions")
    op.drop_index("ix_subscriptions_observed_plan_id", table_name="subscriptions")
    op.drop_column("subscriptions", "entitlement_until")
    op.drop_column("subscriptions", "cancellation_requested_at")
    op.drop_column("subscriptions", "remote_sync")
    op.drop_column("subscriptions", "observed_plan_id")
