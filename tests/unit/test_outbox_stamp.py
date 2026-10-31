"""Outbox payload-stamp statement: typed binds only (asyncpg-safe).

``stamp_outbox_payload_event_id`` runs on the asyncpg driver, which prepares
statements and rejects an untyped parameter feeding ``to_jsonb``'s anyelement
argument — Postgres cannot infer the type of ``$1`` there. The statement must
therefore CAST its binds explicitly, exactly like the uuid binds in
``app/utils/sql_binds.py``. Pinned in source because the failure mode is a
prepare-time database error, not a unit-test-visible one.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.dialects import postgresql

from app.events.writer import stamp_outbox_payload_event_id

_WRITER_SOURCE = Path("src/app/events/writer.py").read_text(encoding="utf-8")


def test_stamp_statement_casts_its_binds() -> None:
    assert "to_jsonb(CAST(:event_id AS text))" in _WRITER_SOURCE
    assert "WHERE id = CAST(:event_id AS uuid)" in _WRITER_SOURCE
    # The untyped shape that fails asyncpg prepare must not reappear.
    assert "to_jsonb(:event_id)" not in _WRITER_SOURCE


def test_stamp_statement_compiles_with_explicit_casts() -> None:
    import inspect

    from sqlalchemy import text

    source = inspect.getsource(stamp_outbox_payload_event_id)
    # The embedded SQL, compiled against the PostgreSQL dialect, renders the
    # CASTs (no bare :event_id placeholder reaches the driver untyped).
    start = source.index("UPDATE outbox_events")
    end = source.index('"""', start)
    rendered = str(text(source[start:end].strip()).compile(dialect=postgresql.dialect()))
    assert "CAST" in rendered
    assert "jsonb_set" in rendered
