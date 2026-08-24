"""Allow re-subscribe after cancellation (Requirement 2.8).

Revision ID: 0004
Revises: 0008
Create Date: 2026-08-17 00:00:00.000000

Replace the hard unique ``(user_id, plan_id)`` constraint with a partial
unique index that only blocks duplicate LIVE subscriptions. CANCELLED /
EXPIRED rows no longer prevent a user from subscribing to the same plan
again.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TERMINAL = "('cancelled', 'expired')"


def upgrade() -> None:
    op.drop_constraint("uq_subscriptions_user_plan", "subscriptions", type_="unique")
    op.execute(
        f"CREATE UNIQUE INDEX uq_subscriptions_user_plan_active "
        f"ON subscriptions (user_id, plan_id) "
        f"WHERE deleted_at IS NULL AND status NOT IN {_TERMINAL}"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_subscriptions_user_plan_active")
    op.create_unique_constraint(
        "uq_subscriptions_user_plan", "subscriptions", ["user_id", "plan_id"]
    )
