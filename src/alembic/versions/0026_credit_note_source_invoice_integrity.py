"""Protect downgrade credit-note source invoices.

Revision ID: 0026
Revises: 0025

Credit notes created by the downgrade workflow must retain the invoice that
funded them.  0024 allowed ``source_invoice_id`` to be NULL and used SET NULL,
which made deleting an invoice silently destroy the economic provenance.  We
cannot invent sources for historical NULL rows, so this migration marks those
rows for manual review, keeps the column staged-nullable, and rejects NULL for
ordinary/new notes at the database check boundary.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0026"
down_revision: str | tuple[str, ...] | None = "0025"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # Preserve historical rows without manufacturing an invoice relation.
    # The marker is queryable and is the explicit exception to the new-note
    # source invariant until an operator resolves the row.
    op.execute(
        sa.text(
            """
            UPDATE credit_notes
            SET metadata_ = jsonb_set(
                COALESCE(metadata_, '{}'::jsonb),
                '{manual_review}',
                'true'::jsonb,
                true
            )
            WHERE source_invoice_id IS NULL
              AND COALESCE(metadata_->>'manual_review', 'false') <> 'true'
            """
        )
    )

    op.drop_constraint("credit_notes_source_invoice_id_fkey", "credit_notes", type_="foreignkey")
    op.create_foreign_key(
        "fk_credit_notes_source_invoice_id",
        "credit_notes",
        "invoices",
        ["source_invoice_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_credit_notes_source_invoice_required",
        "credit_notes",
        "source_invoice_id IS NOT NULL OR (metadata_->>'manual_review') = 'true'",
    )


def downgrade() -> None:
    op.drop_constraint("ck_credit_notes_source_invoice_required", "credit_notes", type_="check")
    op.drop_constraint("fk_credit_notes_source_invoice_id", "credit_notes", type_="foreignkey")
    op.create_foreign_key(
        "credit_notes_source_invoice_id_fkey",
        "credit_notes",
        "invoices",
        ["source_invoice_id"],
        ["id"],
        ondelete="SET NULL",
    )
