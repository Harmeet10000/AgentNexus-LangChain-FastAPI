"""Expression indexes serving the JSONB idempotency-key lookups.

Revision ID: 0027
Revises: 0026
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0027"
down_revision: str | tuple[str, ...] | None = "0026"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

# NOTE: index names/expressions are inlined as literals on purpose — the schema
# identifier gate (`scripts/schema_identifier_gate.py`, asserted by
# tests/unit/test_schema_identifier_gate.py) maps `op.create_index` creators
# from their literal name only, and an index modeled on a `models.Index` must
# have a findable creator. Do not reintroduce a loop over a names table here.

# These serve the per-request `metadata_['key'].as_string() = :param`
# predicates in plans/repository.py (find_by_operation_key,
# find_by_idempotency_scope) and credits/repository.py
# (find_credit_by_operation_key). A plain GIN(metadata_) will NOT serve
# `->>` equality — only an index on the exact expression does.

# Each expression carries its own paren layer — an expression index needs
# DOUBLE parentheses ((expr)); Index(text(...)) in the models emits only one,
# so the extra pair lives here.


def upgrade() -> None:
    op.create_index(
        "ix_plans_billing_operation_key",
        "plans",
        [sa.text("(metadata_ ->> 'billing_operation_key')")],
        unique=False,
    )
    op.create_index(
        "ix_plans_billing_idempotency_scope",
        "plans",
        [sa.text("(metadata_ ->> 'billing_idempotency_scope')")],
        unique=False,
    )
    op.create_index(
        "ix_user_credits_operation_key",
        "user_credits",
        [sa.text("(metadata_ ->> 'operation_key')")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_plans_billing_operation_key", table_name="plans")
    op.drop_index("ix_plans_billing_idempotency_scope", table_name="plans")
    op.drop_index("ix_user_credits_operation_key", table_name="user_credits")
