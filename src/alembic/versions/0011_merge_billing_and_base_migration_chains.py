"""merge billing and base migration chains

Revision ID: 0011
Revises: 0009, 0003
Create Date: 2026-08-21 00:40:15.910025

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | None = ("0009", "0003")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
