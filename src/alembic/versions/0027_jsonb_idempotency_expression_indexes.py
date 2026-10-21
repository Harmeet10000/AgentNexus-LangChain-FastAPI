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

# These serve the per-request `metadata_['key'].as_string() = :param`
# predicates in plans/repository.py (find_by_operation_key,
# find_by_idempotency_scope) and credits/repository.py
# (find_credit_by_operation_key). A plain GIN(metadata_) will NOT serve
# `->>` equality — only an index on the exact expression does.

_EXPRESSION_INDEXES: tuple[tuple[str, str, str], ...] = (
    # NOTE: each expression carries its own paren layer — an expression index
    # needs DOUBLE parentheses ((expr)); CreateTable/Index against
    # Index(text(...)) emits only one, so the extra pair lives here.
    ("ix_plans_billing_operation_key", "plans", "(metadata_ ->> 'billing_operation_key')"),
    ("ix_plans_billing_idempotency_scope", "plans", "(metadata_ ->> 'billing_idempotency_scope')"),
    ("ix_user_credits_operation_key", "user_credits", "(metadata_ ->> 'operation_key')"),
)


def upgrade() -> None:
    for name, table, expression in _EXPRESSION_INDEXES:
        op.create_index(
            name,
            table,
            [sa.text(expression)],
            unique=False,
        )


def downgrade() -> None:
    for name, table, _expression in _EXPRESSION_INDEXES:
        op.drop_index(name, table_name=table)
