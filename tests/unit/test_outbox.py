"""Standalone tests for outbox helper — no shared conftest needed."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock

# Bypass circular import: app.utils.cache.redis_func → app.utils.
# The app.utils module triggers an import cycle on load; this minimal proxy
# prevents that cycle while keeping the import surface explicit via @patch.
_app_utils = types.ModuleType("app.utils")
_app_utils.logger = AsyncMock()
sys.modules["app.utils"] = _app_utils

from app.shared.outbox.helper import OUTBOX_CHANNEL, with_outbox  # noqa: E402 — test import order, fixture dependency


class TestWithOutbox:
    def test_inserts_row_and_notifies(self) -> None:
        calls = []

        async def fake_execute(*args, **kwargs):
            calls.append((args, kwargs))
            return AsyncMock()

        session = AsyncMock()
        session.execute = fake_execute

        event_id = asyncio.run(
            with_outbox(
                session=session,
                aggregate_type="search_document",
                aggregate_id="doc-123",
                event_type="tasks.search_ingest",
                payload={"doc_id": "doc-123"},
            )
        )

        assert event_id is not None
        assert len(calls) == 2

        # First call: INSERT into outbox_events
        insert_sql = calls[0][0][0].text
        insert_params = calls[0][0][1]
        assert "INSERT INTO outbox_events" in insert_sql
        assert insert_params["aggregate_type"] == "search_document"
        assert insert_params["aggregate_id"] == "doc-123"
        assert insert_params["event_type"] == "tasks.search_ingest"
        assert insert_params["id"] == event_id

        # Second call: pg_notify
        notify_sql = calls[1][0][0].text
        notify_params = calls[1][0][1]
        assert "pg_notify" in notify_sql
        assert notify_params["channel"] == OUTBOX_CHANNEL
        assert notify_params["event_id"] == event_id

    def test_rollback_on_exception(self) -> None:
        session = AsyncMock()
        session.execute.side_effect = [AsyncMock(), RuntimeError("pg_notify failed")]

        try:
            asyncio.run(
                with_outbox(
                    session=session,
                    aggregate_type="search_document",
                    aggregate_id="doc-123",
                    event_type="tasks.search_ingest",
                    payload={},
                )
            )
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert "pg_notify failed" in str(exc)
