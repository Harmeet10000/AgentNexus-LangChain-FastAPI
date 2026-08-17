"""Plan versioning support and audit_logs append-only guards.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-17 00:00:00.000000

- Replace the global unique constraint on ``plans.name`` with a partial
  unique index on active plans only, so plan versions (same name, new
  ``parent_plan_id``) can coexist (Requirement 1.6 / 24).
- Add database-level triggers that reject UPDATE/DELETE on ``audit_logs``
  so the audit trail is truly append-only (Requirement 16.5/16.6).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("uq_plans_name", "plans", type_="unique")
    op.execute(
        "CREATE UNIQUE INDEX uq_plans_active_name ON plans (name) WHERE is_active"
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION billing_reject_audit_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'audit_logs is append-only; UPDATE/DELETE is rejected';
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER audit_logs_no_update "
        "BEFORE UPDATE ON audit_logs FOR EACH ROW "
        "EXECUTE FUNCTION billing_reject_audit_mutation()"
    )
    op.execute(
        "CREATE TRIGGER audit_logs_no_delete "
        "BEFORE DELETE ON audit_logs FOR EACH ROW "
        "EXECUTE FUNCTION billing_reject_audit_mutation()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_delete ON audit_logs")
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_update ON audit_logs")
    op.execute("DROP FUNCTION IF EXISTS billing_reject_audit_mutation()")
    op.execute("DROP INDEX IF EXISTS uq_plans_active_name")
    op.create_unique_constraint("uq_plans_name", "plans", ["name"])
