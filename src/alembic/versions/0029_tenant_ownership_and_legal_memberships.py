"""Introduce tenant ownership and tenant-scoped legal memberships.

Existing rows are assigned to a personal tenant (``personal:<user_id>``).
The migration deliberately keeps ``user_id`` on both relations for uploader
attribution and audit, while changing deduplication and ownership predicates
to ``tenant_id``.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0030"
down_revision: str | Sequence[str] | None = "0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "app_tenants",
        sa.Column("id", sa.String(length=255), primary_key=True),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("self_review_allowed", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "app_tenant_memberships",
        sa.Column("tenant_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("legal_role", sa.String(length=32), nullable=False, server_default="analyst"),
        sa.Column(
            "legal_roles",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[\"analyst\"]'"),
        ),
        sa.Column("role_revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", "user_id"),
        sa.ForeignKeyConstraint(["tenant_id"], ["app_tenants.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_app_tenant_memberships_user_id", "app_tenant_memberships", ["user_id"])

    op.add_column("documents", sa.Column("tenant_id", sa.String(length=255), nullable=True))
    op.add_column("chunks", sa.Column("tenant_id", sa.String(length=255), nullable=True))

    # The SQL corpus may contain rows for users which were deleted from Mongo;
    # those rows still receive an isolated personal tenant and remain auditable.
    op.execute(
        """
        INSERT INTO app_tenants (id, display_name, created_at)
        SELECT DISTINCT 'personal:' || user_id, 'Personal tenant ' || user_id, now()
        FROM documents
        ON CONFLICT (id) DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO app_tenant_memberships
            (tenant_id, user_id, legal_role, legal_roles, role_revision, created_at)
        SELECT 'personal:' || user_id, user_id, 'analyst', CAST('["analyst"]' AS JSON), 0, now()
        FROM (SELECT DISTINCT user_id FROM documents) AS source_users
        ON CONFLICT (tenant_id, user_id) DO NOTHING
        """
    )
    op.execute("UPDATE documents SET tenant_id = 'personal:' || user_id WHERE tenant_id IS NULL")
    op.execute(
        """
        UPDATE chunks AS c
        SET tenant_id = d.tenant_id
        FROM documents AS d
        WHERE c.document_id = d.id AND c.tenant_id IS NULL
        """
    )
    op.alter_column("documents", "tenant_id", nullable=False)
    op.alter_column("chunks", "tenant_id", nullable=False)

    op.drop_constraint("uq_documents_user_content_hash", "documents", type_="unique")
    op.create_unique_constraint(
        "uq_documents_tenant_content_hash", "documents", ["tenant_id", "content_hash"]
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])
    op.create_index("ix_chunks_tenant_document", "chunks", ["tenant_id", "document_id"])
    for index_name, kind in (
        ("ix_chunks_kind_legal_contract", "legal_contract"),
        ("ix_chunks_kind_legal_policy", "legal_policy"),
        ("ix_chunks_kind_generic", "generic"),
    ):
        op.execute(f"DROP INDEX IF EXISTS {index_name}")
        op.execute(f"CREATE INDEX {index_name} ON chunks (tenant_id) WHERE chunk_kind = '{kind}'")
    op.execute("DROP INDEX IF EXISTS ix_chunks_instrument_section")
    op.create_index(
        "ix_chunks_instrument_section",
        "chunks",
        ["tenant_id", "instrument_name", "section_ref", "instrument_year"],
        postgresql_where=sa.text("instrument_name IS NOT NULL"),
    )


def downgrade() -> None:
    message = (
        "Tenant ownership migration is irreversible: downgrading would destroy "
        "tenant ownership, membership authority, and audit history. Restore from "
        "a database backup and perform an explicit data migration instead."
    )
    raise RuntimeError(message)
