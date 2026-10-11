"""Standalone tests for outbox helper — no shared conftest needed.

This module used to install a bare `types.ModuleType("app.utils")` into `sys.modules` to dodge the
`app.utils.cache.redis_func -> app.utils` import cycle. Two things are now true and either one on
its own retires the stub: the cycle was severed at source (`319c698`), and the writer
imports nothing beyond `sqlalchemy`, `uuid4`, and its own table model.

Removing it matters beyond tidiness. A bare `ModuleType` has no `__path__`, so while it sat in
`sys.modules` *no* `app.utils.<submodule>` could be imported by any test collected after this one —
`ModuleNotFoundError: ...; 'app.utils' is not a package`, at collection time, which aborts the entire
run. The stub also never restored the real module, so the damage was for the whole session and only
became visible when some later test finally needed a leaf under `app.utils`.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import Insert
from sqlalchemy.dialects import postgresql

from app.events.model import OutboxEvent
from app.events.writer import OUTBOX_CHANNEL, with_outbox


class TestWithOutbox:
    @staticmethod
    def test_inserts_row_and_notifies() -> None:
        calls = []

        async def fake_execute(*args, **kwargs):
            calls.append((args, kwargs))
            return AsyncMock()

        session = AsyncMock()
        session.execute = fake_execute

        event_id = asyncio.run(
            with_outbox(
                session=session,
                aggregate_type="unified_document",
                aggregate_id="doc-123",
                event_type="tasks.documents_ingest",
                payload={"doc_id": "doc-123"},
            )
        )

        assert event_id is not None
        assert len(calls) == 2

        # First call: Core insert against the outbox table. Compared by name,
        # not object identity: the suite imports this module identity twice.
        insert_stmt = calls[0][0][0]
        assert isinstance(insert_stmt, Insert)
        assert insert_stmt.table.name == OutboxEvent.__tablename__
        insert_params = insert_stmt.compile(dialect=postgresql.dialect()).params
        assert insert_params["aggregate_type"] == "unified_document"
        assert insert_params["aggregate_id"] == "doc-123"
        assert insert_params["event_type"] == "tasks.documents_ingest"
        assert insert_params["id"] == event_id
        assert insert_params["payload"] == {"doc_id": "doc-123"}

        # Second call: pg_notify
        notify_sql = calls[1][0][0].text
        notify_params = calls[1][0][1]
        assert "pg_notify" in notify_sql
        assert notify_params["channel"] == OUTBOX_CHANNEL
        assert notify_params["event_id"] == event_id

    @staticmethod
    def test_rollback_on_exception() -> None:
        session = AsyncMock()
        session.execute.side_effect = [AsyncMock(), RuntimeError("pg_notify failed")]

        with pytest.raises(RuntimeError, match="pg_notify failed"):
            asyncio.run(
                with_outbox(
                    session=session,
                    aggregate_type="unified_document",
                    aggregate_id="doc-123",
                    event_type="tasks.documents_ingest",
                    payload={},
                )
            )
