"""Authoritative target shape: outbox, unified documents/chunks, retrieval indexes.

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-22 23:05:00.000000

Defines the whole target shape in one place, forward-only and idempotent.

Why raw ``IF NOT EXISTS`` DDL instead of ORM operations
-------------------------------------------------------
Two load-bearing reasons, not a style preference:

1. ``0008`` is **unstamped and will execute on the next upgrade**,
   creating ``documents`` and ``chunks`` *before* this revision runs.
   Non-idempotent DDL would fail on a duplicate relation.
2. An inspector-based guard needs a **live connection**, which would destroy
   the only proof here that needs no database at all — the offline
   ``--sql`` render of this revision as a scoped range.

Ordering, and why
-----------------
**The outbox relations come first.** They are justified independently of
everything else in this change: they repair two *mounted, public,
rate-limited* endpoints that return 500 today —

- ``POST /auth/forgot-password``   (``features/auth/router.py:195``)
- ``POST /auth/resend-verification`` (``features/auth/router.py:179``)

Both fail in ``AuthService._publish_outbox_event`` (``features/auth/service.py:564``,
called from ``:246`` and ``:273``), which reaches ``with_outbox`` at ``:571``
against an ``outbox_events`` relation that does not exist. They fail **after**
persisting a reset/verification token that no email will ever deliver — a
partial write on shipped surface. Putting them first means a later failure in
this revision cannot block their repair.

Then the extensions, then ``documents`` / ``chunks``, then ``chunks.updated_at``,
then the retrieval indexes.

Extensions are created explicitly, never assumed
------------------------------------------------
All four — ``vector``, ``vectorscale``, ``pg_trgm``, ``pg_textsearch`` —
conditionally, and **before the first dependent object**. The chain's
correctness is otherwise supplied by the *hosting image* rather than by any
revision that will execute. ``vectorscale`` is what provides ``diskann``;
``pg_textsearch`` is what provides ``bm25``.

This makes the dependency **explicit**; it does not make it **portable**, and
the difference matters. Measured against the deployed instance on 2026-08-23,
under a role with ``usesuper = false`` and ``CREATE`` on the database:

===============  =========  =========================================
extension        trusted    consequence for this revision
===============  =========  =========================================
``vector``       false      already installed → ``IF NOT EXISTS`` no-op
``vectorscale``  false      already installed → ``IF NOT EXISTS`` no-op
``pg_textsearch`` false     already installed → ``IF NOT EXISTS`` no-op
``pg_trgm``      **true**   absent, and creatable by this role
===============  =========  =========================================

So three of the four succeed here only because they are *already present*.
``trusted = false`` means a non-superuser cannot install them at all, and
``CREATE EXTENSION IF NOT EXISTS`` does **not** soften that — the guard skips
the work when the extension exists, not when the caller lacks the privilege.
On an image that ships without ``pg_textsearch`` or ``vectorscale`` this
revision therefore **fails** rather than silently degrading, and it fails at
the extension block, before any relation is created. That is the intended
behaviour: the ``bm25`` and ``diskann`` index branches below cannot be
satisfied without them, so surfacing the gap at migration time is strictly
better than a runtime index that matches nothing. It is recorded here so the
failure is recognised as an environment prerequisite and not read as a defect
in this revision.

``pg_trgm`` is the one genuine repair in this block: it is absent from the
deployed instance, it is ``trusted``, and ``0008`` — which executes
before this revision — opens by creating it alongside ``uuid-ossp``. Both were
verified creatable under the deployed role by executing the statement inside a
rolled-back transaction, so the chain's first revision cannot abort on them.

Index names are a query contract
--------------------------------
The keyword-ranking operator takes the index name as a **literal SQL
argument**: ``search_text <@> to_bm25query(:query, 'chunks_bm25_idx')``. An
index of the right shape under a different name **matches nothing and reports
no error**. The binding contract for these relations is therefore the literal
``'chunks_bm25_idx'``, which appears six times in
``features/documents/repository.py`` (``:331``, ``:335``, ``:339``, ``:543``, …)
and matches the name ``0008`` already declares. Every retrieval index
below is created under the exact name that revision uses, so the two converge
instead of producing two differently-named indexes of the same shape.

Note for the reader: ``features/search/constants.py:15`` pins
``SEARCH_CHUNKS_BM25_INDEX_NAME = "search_chunks_bm25_idx"``. That constant has
**zero readers**, and it names an index on the *separate* ``search_chunks``
relation, not on ``chunks``. It is not the contract for this revision.

``chunks.updated_at`` carries a server default, deliberately
-----------------------------------------------------------
When this revision was written, neither the ``UnifiedChunk`` ORM model
(``features/documents/model.py:73`` — the class is *not* named ``Chunk``; that
name belongs to an unrelated Pydantic model in
``shared/rag/document_processing/models.py``) nor any query in the repository
declared or read ``updated_at``. Only ``UnifiedDocument`` had one. A ``NOT NULL``
column with no default would therefore break every ORM insert, because nothing
supplies a value. ``DEFAULT now()`` keeps it satisfiable from the database side
alone.

That reasoning was correct and incomplete. "The ORM does not declare it" is also
exactly the condition under which autogenerate proposes to **DROP** the column —
which was then measured: ``alembic check`` reported ``remove_column`` for
``chunks.updated_at`` and ``remove_index`` for all three retrieval indexes
below, plus ``idx_outbox_unpublished``. Adding an object in raw DDL without a
matching registry declaration trades a missing object for a droppable one.

All six are now declared on the models, with ``server_default=func.now()`` here
so the ORM and this DDL agree textually (``env.py`` sets
``compare_server_default=True``). The column keeps its server default; the
declaration is what makes it safe.

Reversal
--------
``downgrade()`` is an intentional no-op. See ``0012`` — reversal below
that merge is unsupported, and dropping these relations here would re-break
the two public endpoints above while leaving ``0006`` and ``0008``
still claiming to have created them.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0013"
down_revision: str | Sequence[str] | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Outbox relations first — repairs two public endpoints returning 500.
    #    No extension dependency, so nothing can block this half.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS outbox_events (
            id               varchar(36)  NOT NULL,
            aggregate_type   varchar(64)  NOT NULL,
            aggregate_id     varchar(128) NOT NULL,
            event_type       varchar(64)  NOT NULL,
            payload          jsonb        NOT NULL,
            created_at       timestamptz  NOT NULL,
            published_at     timestamptz,
            publish_attempts integer      NOT NULL DEFAULT 0,
            last_error       text,
            CONSTRAINT pk_outbox_events PRIMARY KEY (id)
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_outbox_unpublished
            ON outbox_events (created_at)
            WHERE published_at IS NULL
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS dead_letter_events (
            id                varchar(36)  NOT NULL,
            original_event_id varchar(36)  NOT NULL,
            aggregate_type    varchar(64)  NOT NULL,
            aggregate_id      varchar(128) NOT NULL,
            event_type        varchar(64)  NOT NULL,
            payload           jsonb        NOT NULL,
            created_at        timestamptz  NOT NULL,
            dead_letter_at    timestamptz  NOT NULL,
            last_error        text         NOT NULL,
            CONSTRAINT pk_dead_letter_events PRIMARY KEY (id)
        )
    """)

    # ------------------------------------------------------------------
    # 2. Extensions — all four, conditionally, before the first dependent
    #    object. vectorscale provides diskann; pg_textsearch provides bm25.
    # ------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")

    # ------------------------------------------------------------------
    # 3. Unified documents / chunks.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id            uuid         NOT NULL,
            user_id       varchar(255) NOT NULL,
            title         varchar(500) NOT NULL,
            source_uri    text,
            object_uri    text         NOT NULL,
            content_hash  varchar(64)  NOT NULL,
            document_kind varchar(64)  NOT NULL,
            status        varchar(64)  NOT NULL,
            jurisdiction  varchar(255),
            contract_type varchar(255),
            parties       jsonb        NOT NULL,
            metadata_     jsonb        NOT NULL,
            created_at    timestamptz  NOT NULL,
            updated_at    timestamptz  NOT NULL,
            CONSTRAINT pk_documents PRIMARY KEY (id),
            CONSTRAINT uq_documents_user_content_hash UNIQUE (user_id, content_hash)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_user_id ON documents (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_kind ON documents (document_kind)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_status ON documents (status)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_metadata_gin ON documents USING gin (metadata_)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id                  uuid         NOT NULL,
            document_id         uuid         NOT NULL,
            user_id             varchar(255) NOT NULL,
            chunk_index         integer      NOT NULL,
            chunk_kind          varchar(64)  NOT NULL,
            content             text         NOT NULL,
            preamble            text         NOT NULL,
            clause_type         varchar(128),
            page_no             integer      NOT NULL,
            embedding           vector(768),
            metadata_           jsonb        NOT NULL,
            custom_metadata     jsonb        NOT NULL,
            quality_warnings    jsonb        NOT NULL,
            graphiti_episode_id varchar(255),
            graphiti_verified   boolean      NOT NULL,
            search_text         text         GENERATED ALWAYS AS (
                                                 COALESCE(clause_type, '') || ' ' ||
                                                 COALESCE(preamble, '')    || ' ' ||
                                                 COALESCE(content, '')
                                             ) STORED NOT NULL,
            created_at          timestamptz  NOT NULL,
            CONSTRAINT pk_chunks PRIMARY KEY (id),
            CONSTRAINT uq_chunks_document_chunk_index UNIQUE (document_id, chunk_index),
            CONSTRAINT fk_chunks_document_id_documents FOREIGN KEY (document_id)
                REFERENCES documents (id) ON DELETE CASCADE
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_user_document ON chunks (user_id, document_id)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_kind ON chunks (chunk_kind)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_graphiti_verified ON chunks (graphiti_verified)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_metadata_gin ON chunks USING gin (metadata_)")

    # ------------------------------------------------------------------
    # 4. chunks.updated_at — absent from both the DDL and the ORM, so it
    #    must be satisfiable without an application-supplied value.
    # ------------------------------------------------------------------
    op.execute("""
        ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now()
    """)

    # ------------------------------------------------------------------
    # 5. Retrieval indexes — all three branches, under the exact names the
    #    query layer passes as literal SQL arguments.
    # ------------------------------------------------------------------
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_bm25_idx
            ON chunks USING bm25 (search_text)
            WITH (text_config='english', k1=1.2, b=0.75)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING diskann (embedding vector_cosine_ops)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_search_text_trgm_idx
            ON chunks USING gin (search_text gin_trgm_ops)
    """)


def downgrade() -> None:
    """Intentional no-op — see the module docstring's Reversal section."""
