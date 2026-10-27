"""Persist tenant mirror retry state and canonicalize legacy legal roles."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0033"
down_revision: str | None = "0032"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # Contract identity is nullable by design: legacy rows are explicitly
    # unknown rather than fabricated as contract zero.
    op.add_column("chunks", sa.Column("contract_index", sa.Integer(), nullable=True))
    op.add_column(
        "app_tenant_memberships",
        sa.Column("mirror_sync_pending", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "app_tenant_memberships",
        sa.Column("mirror_sync_error", sa.String(length=500), nullable=True),
    )
    # ``admin`` was the pre-tenant legal-role spelling.  Keep reads tolerant
    # in application code, but canonicalize durable SQL values at the boundary.
    op.execute(
        "UPDATE app_tenant_memberships SET legal_role = 'corpus_admin' WHERE legal_role = 'admin'"
    )
    op.execute(
        "UPDATE app_tenant_memberships SET legal_roles = "
        "CAST(REPLACE(CAST(legal_roles AS TEXT), '\"admin\"', '\"corpus_admin\"') AS JSON) "
        "WHERE CAST(legal_roles AS TEXT) LIKE '%\"admin\"%'"
    )


def downgrade() -> None:
    op.drop_column("chunks", "contract_index")
    message = (
        "Tenant mirror reconciliation migration is irreversible; pending-sync "
        "history must not be discarded by downgrade."
    )
    raise RuntimeError(message)
