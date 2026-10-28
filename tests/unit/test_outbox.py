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
from app.events.writer import (
    OUTBOX_CHANNEL,
    PostgresOutboxAdapter,
    terminal_event_id,
    with_outbox,
)
from app.shared.jobs import AggregateRef, InMemoryDirectAdapter, JobBus, TypedJob
from app.shared.jobs.catalog import JOB_CATALOG
from app.shared.jobs.names import CRAWLER_CRAWL, DOCUMENTS_INGEST
from app.shared.jobs.registry import TaskPayloadValidationError, UnregisteredTaskError

_DOCUMENT_PAYLOAD = {
    "document_id": "doc-123",
    "user_id": "user-123",
    "filename": "contract.pdf",
    "content_type": "application/pdf",
    "object_uri": "s3://documents/contract.pdf",
}


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
                event_type=DOCUMENTS_INGEST,
                payload=_DOCUMENT_PAYLOAD,
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
        assert insert_params["payload"] == _DOCUMENT_PAYLOAD

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
                    event_type=DOCUMENTS_INGEST,
                    payload=_DOCUMENT_PAYLOAD,
                )
            )


def test_terminal_event_identity_is_stable_but_attempt_scoped() -> None:
    payload = {
        "tenant_id": "tenant-a",
        "document_id": "doc-123",
        "document_version": 2,
        "status": "failed",
        "attempt": 1,
        "task_id": "task-a",
    }

    assert terminal_event_id(payload) == terminal_event_id(dict(payload))
    assert terminal_event_id(payload | {"task_id": "task-b"}) != terminal_event_id(payload)
    assert terminal_event_id(payload | {"document_version": 3}) != terminal_event_id(payload)


@pytest.mark.asyncio
async def test_transactional_job_bus_stages_on_same_session_without_owning_lifecycle() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=AsyncMock())
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()

    job = JOB_CATALOG[DOCUMENTS_INGEST].prepare(_DOCUMENT_PAYLOAD)
    bus = JobBus(InMemoryDirectAdapter(), PostgresOutboxAdapter())

    receipt = await bus.in_transaction(session).enqueue(
        job,
        aggregate=AggregateRef(aggregate_type="user_document", aggregate_id="doc-123"),
    )

    assert receipt.event_id
    assert session.execute.await_count == 2
    insert_call, notify_call = session.execute.await_args_list
    insert_stmt = insert_call.args[0]
    insert_params = insert_stmt.compile(dialect=postgresql.dialect()).params
    assert insert_params["event_type"] == DOCUMENTS_INGEST
    assert insert_params["payload"] == _DOCUMENT_PAYLOAD
    assert insert_params["id"] == receipt.event_id
    assert notify_call.args[1]["event_id"] == receipt.event_id
    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()
    session.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_transactional_job_is_rejected_before_sql() -> None:
    session = AsyncMock()
    session.execute = AsyncMock()
    invalid = TypedJob(
        definition=JOB_CATALOG[DOCUMENTS_INGEST],
        payload=JOB_CATALOG[CRAWLER_CRAWL].payload_model(
            crawl_id="crawl-123", request_payload={"url": "https://example.com"}
        ),
    )
    bus = JobBus(InMemoryDirectAdapter(), PostgresOutboxAdapter())

    with pytest.raises(TaskPayloadValidationError):
        await bus.in_transaction(session).enqueue(
            invalid,
            aggregate=AggregateRef(aggregate_type="user_document", aggregate_id="doc-123"),
        )

    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_facade_rejects_unknown_job_before_sql() -> None:
    session = AsyncMock()

    with pytest.raises(UnregisteredTaskError):
        await with_outbox(
            session=session,
            aggregate_type="user_document",
            aggregate_id="doc-123",
            event_type="tasks.not_in_catalog",
            payload={},
        )

    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_sql_failure_propagates_without_transaction_ownership() -> None:
    session = AsyncMock()
    failure = RuntimeError("database unavailable")
    session.execute = AsyncMock(side_effect=failure)
    session.rollback = AsyncMock()

    job = JOB_CATALOG[DOCUMENTS_INGEST].prepare(_DOCUMENT_PAYLOAD)
    bus = JobBus(InMemoryDirectAdapter(), PostgresOutboxAdapter())

    with pytest.raises(RuntimeError, match="database unavailable"):
        await bus.in_transaction(session).enqueue(
            job,
            aggregate=AggregateRef(aggregate_type="user_document", aggregate_id="doc-123"),
        )

    session.rollback.assert_not_awaited()
