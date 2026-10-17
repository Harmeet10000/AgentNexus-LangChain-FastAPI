"""Agent Saul transcript persistence: session status cache and message projection keys.

Revision ID: 0022
Revises: 0021

Precondition: ``chat_sessions`` and ``chat_messages`` MUST be empty. This revision
backfills nothing and cannot; sessions written before the projector existed have
no turn_seq / ordinal / source_key lineage. The upgrade aborts with a clear
message when data is present — re-provisioning chat tables (or a separate
backfill change) is the operator's call, not this revision's.

LangGraph's ``checkpoints`` / ``checkpoint_blobs`` / ``checkpoint_writes`` /
``checkpoint_migrations`` tables are NEVER referenced here, in either direction.
Autogenerate excludes them via ``env.include_object`` (reflected, no injected
counterpart); this revision deliberately adds nothing that touches them either.

Design: ``docs/agent-saul-chat-transcript-persistence.md`` §3, §11.
"""

from __future__ import annotations

from sqlalchemy import text

from alembic import op

revision: str = "0022"
down_revision: str | tuple[str, ...] | None = "0021"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

_STATUS_ENUM = "chat_session_status"

_KIND_CHECK = (
    "kind IN ('query', 'clarification_prompt', 'clarification_reply', "
    "'plan_prompt', 'plan_decision', 'final_summary', 'turn_closed')"
)


def _row_count(connection: object, table: str) -> int:
    """Rows in ``table``; 0 when the table does not exist (fresh databases)."""
    existing = connection.scalar(
        text(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = :table"
        ),
        {"table": table},
    )
    if not existing:
        return 0
    result = connection.scalar(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
    return int(result or 0)


def upgrade() -> None:
    conn = op.get_bind()

    session_rows = _row_count(conn, "chat_sessions")
    message_rows = _row_count(conn, "chat_messages")
    if session_rows or message_rows:
        msg = (
            "0022 requires empty chat tables: chat_sessions has "
            f"{session_rows} row(s), chat_messages has {message_rows} row(s). "
            "The transcript projection keys (turn_seq, ordinal, source_key) "
            "cannot be backfilled by this revision; purge the chat tables or "
            "write an explicit backfill change first."
        )
        raise RuntimeError(msg)

    # 0. Session-status cache enum. Same DO-block pattern as message_role (0014).
    op.execute(f"""
        DO $$
        BEGIN
            CREATE TYPE {_STATUS_ENUM} AS ENUM (
                'active', 'clarifying', 'resumed',
                'completed', 'abandoned_expired', 'failed'
            );
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
    """)

    # 1. chat_sessions: projection cache columns.
    #    status is a *cache* of the latest turn's derived status; the truth
    #    lives in the checkpoint (the projector, §4.1). server_default keeps
    #    the column NOT NULL without a backfill.
    op.execute("""
        ALTER TABLE chat_sessions
            ADD COLUMN schema_version       smallint              NOT NULL DEFAULT 1,
            ADD COLUMN status               chat_session_status   NOT NULL DEFAULT 'active',
            ADD COLUMN turn_count           integer               NOT NULL DEFAULT 0,
            ADD COLUMN last_checkpoint_id   text,
            ADD COLUMN last_checkpoint_ts   timestamptz
    """)

    # 2. chat_messages: projection lineage columns. The table is empty, so NOT
    #    NULL carries no backfill cost.
    op.execute("""
        ALTER TABLE chat_messages
            ADD COLUMN turn_seq             integer   NOT NULL,
            ADD COLUMN ordinal              integer   NOT NULL,
            ADD COLUMN kind                 varchar(32) NOT NULL,
            ADD COLUMN source_key           text      NOT NULL,
            ADD COLUMN source_checkpoint_id text      NOT NULL,
            ADD COLUMN redacted_at          timestamptz
    """)

    # 3. Erasure stores NULL content ("no content"), not the empty string
    #    ("empty content that was erased") — doc §3.2 [v2].
    op.execute("ALTER TABLE chat_messages ALTER COLUMN content DROP NOT NULL")

    # 4. Timestamps get the project's server_default (database/base.py:
    #    TimestampMixin rationale — DB clock, not app clocks).
    op.execute("ALTER TABLE chat_sessions ALTER COLUMN created_at SET DEFAULT now()")
    op.execute("ALTER TABLE chat_sessions ALTER COLUMN updated_at SET DEFAULT now()")
    op.execute("ALTER TABLE chat_messages ALTER COLUMN created_at SET DEFAULT now()")

    # 5. kind CHECK — a constraint, not a PG enum: kinds will evolve, and the
    #    CHECK is reversible without ALTER TYPE surgery (doc §3.2).
    op.execute(f"""
        ALTER TABLE chat_messages
            ADD CONSTRAINT ck_chat_messages_kind CHECK ({_KIND_CHECK})
    """)

    # 6. Idempotency guard: one transcript row per projected item.
    op.execute("""
        ALTER TABLE chat_messages
            ADD CONSTRAINT uq_chat_messages_session_source_key
            UNIQUE (session_id, source_key)
    """)

    # 7. Read-order indexes (D3): display order is (turn_seq, ordinal), not
    #    insert order, because reconciliation inserts late.
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_chat_messages_session_turn
            ON chat_messages (session_id, turn_seq, ordinal)
    """)

    # 8. The chat list is by last activity; the old single-column
    #    ix_chat_sessions_user_id is a redundant prefix of this composite.
    op.execute("DROP INDEX IF EXISTS ix_chat_sessions_user_id")
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_updated
            ON chat_sessions (user_id, updated_at DESC)
    """)

    print(  # noqa: T201 — operational sanity output requested by design §13 DoD
        "0022 upgrade ok: agent-saul transcript projection columns applied; "
        "chat tables empty precondition held; checkpoint* tables untouched"
    )


def downgrade() -> None:
    conn = op.get_bind()
    session_rows = _row_count(conn, "chat_sessions")
    message_rows = _row_count(conn, "chat_messages")
    if session_rows or message_rows:
        print(  # noqa: T201
            f"0022 downgrade warning: chat_sessions={session_rows} "
            f"chat_messages={message_rows} row(s) present; downgrade will "
            "DROP the transcript projection columns and their data"
        )

    op.execute("DROP INDEX IF EXISTS ix_chat_sessions_user_updated")
    op.execute("CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_id ON chat_sessions (user_id)")
    op.execute("DROP INDEX IF EXISTS ix_chat_messages_session_turn")
    op.execute(
        "ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS uq_chat_messages_session_source_key"
    )
    op.execute("ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS ck_chat_messages_kind")
    op.execute("ALTER TABLE chat_messages ALTER COLUMN content SET NOT NULL")
    for column in (
        "redacted_at",
        "source_checkpoint_id",
        "source_key",
        "kind",
        "ordinal",
        "turn_seq",
    ):
        op.execute(f"ALTER TABLE chat_messages DROP COLUMN IF EXISTS {column}")
    op.execute("UPDATE chat_sessions SET status = 'active'")  # collapse cache before type drop
    for column in (
        "last_checkpoint_ts",
        "last_checkpoint_id",
        "turn_count",
        "status",
        "schema_version",
    ):
        op.execute(f"ALTER TABLE chat_sessions DROP COLUMN IF EXISTS {column}")
    op.execute(f"DROP TYPE IF EXISTS {_STATUS_ENUM}")
    print(  # noqa: T201 — operational sanity output
        "0022 downgrade ok: transcript projection columns removed; checkpoint* tables untouched"
    )
