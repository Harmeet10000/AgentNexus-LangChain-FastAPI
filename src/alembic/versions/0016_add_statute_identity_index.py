"""Serve the statute point lookup from an index

The legal-corpus-retrieval capability requires a lookup on the statute identity
attributes to be index-served rather than scanned. The attributes themselves
landed in revision 0015; this revision adds the btree that answers
``WHERE instrument_name = … AND section_ref = … ORDER BY instrument_year DESC NULLS LAST``
without touching every chunk row (NULL years sort after dated versions).

Applying DDL to any deployed instance remains a separately authorized act;
this revision ships unexecuted until then.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0016"
down_revision: str | tuple[str, ...] | None = "0015"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_chunks_instrument_section",
        "chunks",
        ["instrument_name", "section_ref", sa.text("instrument_year DESC NULLS LAST")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_instrument_section", table_name="chunks")
