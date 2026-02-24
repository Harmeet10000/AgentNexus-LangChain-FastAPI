"""rename_metadata_to_meta_data

Revision ID: 2bc7726317f6
Revises: c0c17c6eb1cc
Create Date: 2026-02-24 21:00:12.262132

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '2bc7726317f6'
down_revision: str | None = 'c0c17c6eb1cc'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column('document_vectors', 'metadata', new_column_name='meta_data')


def downgrade() -> None:
    op.alter_column('document_vectors', 'meta_data', new_column_name='metadata')
