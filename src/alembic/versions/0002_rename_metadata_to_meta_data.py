"""rename_metadata_to_meta_data

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-24 21:00:12.262132

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("document_vectors", "metadata", new_column_name="meta_data")


def downgrade() -> None:
    op.alter_column("document_vectors", "meta_data", new_column_name="metadata")
