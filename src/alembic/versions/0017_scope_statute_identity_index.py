"""Scope the statute point-lookup index to its tenant with a partial predicate

Revision 0016 shipped ``ix_chunks_instrument_section`` as a full index on
``(instrument_name, section_ref, instrument_year)``. The design and ADR
require the tenant-scoped partial form on
``(user_id, instrument_name, section_ref, instrument_year)``
``WHERE instrument_name IS NOT NULL``: every read in this schema is
tenant-scoped, so an index that does not lead with ``user_id`` is not
usable by any permitted read, and the partial predicate keeps the
non-statute majority of chunks off the index entirely.

This revision drops the 0016 index and creates the scoped partial one
under the same name. Column order is load-bearing: tenant first, the two
identity columns next (one index descent for the point lookup), year
last so a backward scan yields the newest applicable vintage first
without a sort.

Applying DDL to any deployed instance remains a separately authorized act;
this revision ships unexecuted until then.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0017"
down_revision: str | tuple[str, ...] | None = "0016"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.drop_index("ix_chunks_instrument_section", table_name="chunks")
    op.create_index(
        "ix_chunks_instrument_section",
        "chunks",
        ["user_id", "instrument_name", "section_ref", sa.text("instrument_year DESC NULLS LAST")],
        unique=False,
        postgresql_where=sa.text("instrument_name IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_instrument_section", table_name="chunks")
    op.create_index(
        "ix_chunks_instrument_section",
        "chunks",
        ["instrument_name", "section_ref", sa.text("instrument_year DESC NULLS LAST")],
        unique=False,
    )
