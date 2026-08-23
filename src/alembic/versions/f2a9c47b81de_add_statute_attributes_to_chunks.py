"""Add the three statute attributes to chunks and settle updated_at ownership

Why this revision exists

Step 15 of documents-unified-schema planned to add ``instrument_name``,
``section_ref`` and ``instrument_year`` to the ORM with no accompanying DDL, on
the stated basis that change 0's ``CREATE TABLE chunks`` ships them. Measured
against migration history it does not: no revision creates these columns, so an
ORM-only addition would have been invisible divergence -- the identifier gate
checks index and constraint names, not columns.

What it does

1. Adds the three statute attributes as nullable columns. Nothing in this
   change populates them; change 3's legal-corpus-retrieval is their reader.
2. Drops the server default on ``chunks.updated_at``. Ownership is settled as
   ORM/application-side, matching ``documents.updated_at``: the application's
   row builder supplies the timestamp and the chunk upsert's conflict set
   refreshes it explicitly. The previous arrangement left every chunk's
   ``updated_at`` equal to its creation time forever, because SQLAlchemy does
   not merge ``onupdate`` defaults into an explicit ``DO UPDATE SET``.

Applying DDL to any deployed instance remains a separately authorized act;
this revision ships unexecuted until then.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "f2a9c47b81de"
down_revision: str | tuple[str, ...] | None = "b3e7c41d92af"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("instrument_name", sa.String(length=255), nullable=True))
    op.add_column("chunks", sa.Column("section_ref", sa.String(length=255), nullable=True))
    op.add_column("chunks", sa.Column("instrument_year", sa.Integer(), nullable=True))
    # The ORM owns this default now (mirrors documents.updated_at); change 0's
    # ADD COLUMN carried DEFAULT now(), which duplicates an application-side
    # value the row builder already supplies.
    op.execute("ALTER TABLE chunks ALTER COLUMN updated_at DROP DEFAULT")


def downgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN updated_at SET DEFAULT now()")
    op.drop_column("chunks", "instrument_year")
    op.drop_column("chunks", "section_ref")
    op.drop_column("chunks", "instrument_name")
