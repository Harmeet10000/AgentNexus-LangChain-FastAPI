"""Scope site-search tables by tenant and track ingest status.

The site-search tables were created in 0004 without tenancy — the legacy
feature had no authentication at all. This revision adds the tenant mirror
the modern feature requires:

- ``search_documents.tenant_id`` / ``search_chunks.tenant_id`` (indexed, NOT
  NULL after backfill), so every read and write is tenant-scoped and the
  content-hash dedupe becomes per tenant. Pre-tenancy rows are assigned to a
  dedicated ``search:legacy`` tenant that no caller can authenticate into,
  which fails closed: legacy rows stay addressable for auditing but invisible
  to every search.
- ``search_documents.status`` for the ingest pipeline lifecycle.

It also removes the legacy ``search_chunks.content_tsv`` generated tsvector
column and its GIN index. The lexical leg is pg_textsearch's bm25 access
method directly on ``content`` (see ``search_repository.bm25_search``); the
tsvector would double-count that signal in the three-branch fusion, and
nothing in the application ever read it.

Two database shapes reach this revision (idempotent against both)
---------------------------------------------------------------
- 0004-executed databases: the search tables carry the full index set.
- 0014-repaired databases (stamped, never migrated): ``0014`` recreates the
  tables with **no indexes at all** — its docstring records that
  ``search_chunks_bm25_idx`` was referenced by settings but never created.

Therefore every index drop here is ``DROP INDEX IF EXISTS`` (a plain
``op.drop_index`` would abort on the 0014 shape), and the retrieval indexes
the search branches require — bm25, DiskANN, trigram GIN, metadata GIN,
document-id btree, plus their extensions — are created ``IF NOT EXISTS`` with
the exact definitions/names 0004 used and the ORM declares, so search works
after either upgrade path.

Revision ID: 0034
Revises: 0033
Create Date: 2026-10-31 00:00:00.000000
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "0034"
down_revision: str | Sequence[str] | None = "0033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Tenant assigned to rows that predate tenancy. No authenticated caller maps
#: to it, so tenant-scoped queries exclude the legacy corpus by construction.
#: Referenced by the inlined literal in ``upgrade``; kept named so the value
#: has exactly one spelled-out home.
_LEGACY_TENANT = "search:legacy"

#: The retrieval indexes the search branches require, in the exact shape 0004
#: created them on executed databases. Applied ``IF NOT EXISTS`` so the
#: 0014-repaired shape (tables, no indexes) gains them and the 0004 shape
#: keeps its originals untouched.
_SEARCH_RETRIEVAL_INDEXES = (
    # Lexical leg: pg_textsearch bm25 directly on content. The name is a
    # literal contract — search_repository passes it to to_bm25query().
    """
    CREATE INDEX IF NOT EXISTS search_chunks_bm25_idx ON search_chunks
    USING bm25(content) WITH (text_config='english')
    """,
    # Dense leg: DiskANN cosine over the 768-dim embeddings.
    """
    CREATE INDEX IF NOT EXISTS search_chunks_embedding_idx ON search_chunks
    USING diskann (embedding vector_cosine_ops)
    """,
    # Fuzzy leg: pg_trgm GIN over content.
    """
    CREATE INDEX IF NOT EXISTS ix_search_chunks_content_trgm ON search_chunks
    USING gin (content gin_trgm_ops)
    """,
    # Metadata JSONB containment filters.
    """
    CREATE INDEX IF NOT EXISTS ix_search_chunks_chunk_metadata_gin ON search_chunks
    USING gin (chunk_metadata)
    """,
    # Foreign-key btree for hydration joins and cascade deletes.
    "CREATE INDEX IF NOT EXISTS ix_search_chunks_document_id ON search_chunks (document_id)",
)


def upgrade() -> None:
    # The 0014-repaired shape never ran 0004's extension setup either.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    op.add_column(
        "search_documents",
        sa.Column("tenant_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "search_documents",
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
    )
    op.add_column(
        "search_chunks",
        sa.Column("tenant_id", sa.String(length=255), nullable=True),
    )

    # Pre-tenancy rows belong to no caller. Pin them to an unmatchable system
    # tenant (``search:legacy``, the _LEGACY_TENANT value, inlined because a
    # migration constant is not a bindable parameter) rather than guessing an
    # owner; chunks inherit their document's tenant.
    op.execute("UPDATE search_documents SET tenant_id = 'search:legacy' WHERE tenant_id IS NULL")
    op.execute(
        """
        UPDATE search_chunks AS c
        SET tenant_id = d.tenant_id
        FROM search_documents AS d
        WHERE c.document_id = d.id AND c.tenant_id IS NULL
        """
    )
    op.alter_column("search_documents", "tenant_id", nullable=False)
    op.alter_column("search_chunks", "tenant_id", nullable=False)

    # Documents that already carried chunks completed the legacy pipeline;
    # everything else stays in the pre-pipeline state its default assigns.
    op.execute(
        """
        UPDATE search_documents AS d
        SET status = 'completed'
        WHERE EXISTS (SELECT 1 FROM search_chunks AS c WHERE c.document_id = d.id)
        """
    )

    # Dedupe was global; it must be per tenant, or one tenant's ingest would
    # collide with another's identical content.
    op.drop_constraint("uq_search_documents_content_hash", "search_documents", type_="unique")
    op.create_unique_constraint(
        "uq_search_documents_tenant_content_hash",
        "search_documents",
        ["tenant_id", "content_hash"],
    )

    op.create_index("ix_search_documents_tenant_id", "search_documents", ["tenant_id"])
    op.create_index("ix_search_chunks_tenant_id", "search_chunks", ["tenant_id"])

    # The tsvector column is unread legacy surface: pg_textsearch bm25 is the
    # lexical leg. Dropping it (and its GIN index) removes the double-count
    # hazard and the maintenance cost of the GENERATED ALWAYS column. The drop
    # is guarded because only the 0004 shape ever had either.
    op.execute("DROP INDEX IF EXISTS ix_search_chunks_content_tsv_gin")
    op.drop_column("search_chunks", "content_tsv")

    for create_statement in _SEARCH_RETRIEVAL_INDEXES:
        op.execute(create_statement)


def downgrade() -> None:
    # Restoring the GLOBAL content-hash unique constraint is only possible
    # while the corpus is still single-tenant per hash. Normal cross-tenant
    # operation makes identical content legal per tenant, so a naive downgrade
    # would fail mid-constraint-swap or, worse, tempt a "fix" that deletes
    # tenant data. Policy: fail loudly, name the conflicting hashes, change
    # nothing. The operator must resolve duplicates explicitly (delete or
    # re-key per tenant) before downgrading.
    conn = op.get_bind()
    conflicts = conn.execute(
        sa.text(
            """
            SELECT content_hash, COUNT(DISTINCT tenant_id) AS tenant_count
            FROM search_documents
            GROUP BY content_hash
            HAVING COUNT(DISTINCT tenant_id) > 1
            ORDER BY content_hash
            LIMIT 25
            """
        )
    ).fetchall()
    if conflicts:
        listing = ", ".join(f"{row.content_hash} ({row.tenant_count} tenants)" for row in conflicts)
        msg = (
            "Cannot downgrade 0034: search_documents contains content hashes "
            "owned by more than one tenant, which the global "
            f"uq_search_documents_content_hash constraint forbids: {listing}. "
            "Resolve the duplicates (delete or re-key per tenant) and retry; "
            "this downgrade never deletes tenant data automatically."
        )
        raise RuntimeError(msg)

    op.add_column(
        "search_chunks",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
    )
    # GIN, matching 0004's original — a plain btree would not serve the
    # tsvector containment lookups the legacy shape relied on.
    op.execute(
        "CREATE INDEX ix_search_chunks_content_tsv_gin ON search_chunks USING gin (content_tsv)"
    )
    op.drop_index("ix_search_chunks_tenant_id", table_name="search_chunks")
    op.drop_index("ix_search_documents_tenant_id", table_name="search_documents")
    op.drop_constraint(
        "uq_search_documents_tenant_content_hash", "search_documents", type_="unique"
    )
    op.create_unique_constraint(
        "uq_search_documents_content_hash",
        "search_documents",
        ["content_hash"],
    )
    op.drop_column("search_chunks", "tenant_id")
    op.drop_column("search_documents", "status")
    op.drop_column("search_documents", "tenant_id")
