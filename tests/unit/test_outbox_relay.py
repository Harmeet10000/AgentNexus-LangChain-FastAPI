"""Outbox relay collaborator ownership and graceful-drain behavior."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, cast

import asyncpg_listen
import pytest

import app.events.relay as relay_module
from app.events.relay import OutboxRelay
from app.events.writer import terminal_event_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    import asyncpg

    from app.shared.jobs import EnqueueOptions, TypedJob


_PAYLOAD = {
    "document_id": "doc-1",
    "user_id": "user-1",
    "filename": "contract.pdf",
    "content_type": "application/pdf",
    "object_uri": "s3://documents/contract.pdf",
}


class _DirectPublisher:
    def __init__(self, *, result: object = "task-1") -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.result = result

    async def publish(self, job: TypedJob[Any], *, options: EnqueueOptions) -> object:
        _ = options
        self.calls.append((job.task_name, job.wire_payload()))
        return self.result


class _MappingResult:
    def __init__(self, row: dict[str, object] | None) -> None:
        self._row = row

    def mappings(self) -> _MappingResult:
        return self

    def one_or_none(self) -> dict[str, object] | None:
        return self._row

    def all(self) -> list[dict[str, object]]:
        return [] if self._row is None else [self._row]


class _ScalarResult:
    def __init__(self, value: str) -> None:
        self.value = value

    def scalar_one_or_none(self) -> str:
        return self.value


class _DeadLetterSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object] | None]] = []
        self.commits = 0

    async def execute(
        self, statement: Any, parameters: dict[str, object] | None = None
    ) -> _MappingResult | _ScalarResult:
        sql = str(statement)
        self.calls.append((sql, parameters))
        if "FOR UPDATE" in sql:
            return _MappingResult(
                {
                    "publish_attempts": 4,
                    "event_type": "tasks.documents_ingest",
                    "aggregate_id": "doc-1",
                    "payload": {
                        "document_id": "doc-1",
                        "user_id": "user-1",
                        "tenant_id": "tenant-1",
                        "document_version": 2,
                    },
                }
            )
        if "INSERT INTO outbox_events" in sql:
            return _ScalarResult("terminal-event")
        return _MappingResult(None)

    async def commit(self) -> None:
        self.commits += 1


class _Session:
    def __init__(self, *, pause_update: bool = False) -> None:
        self.pause_update = pause_update
        self.update_started = asyncio.Event()
        self.release_update = asyncio.Event()
        self.commits = 0
        self.statements: list[str] = []

    async def execute(
        self, statement: Any, _parameters: dict[str, object] | None = None
    ) -> _MappingResult:
        sql = str(statement)
        self.statements.append(sql)
        if "outbox_events" in sql and "FOR UPDATE" in sql:
            return _MappingResult(
                {
                    "id": "event-1",
                    "event_type": "tasks.documents_ingest",
                    "payload": {
                        "document_id": "doc-1",
                        "user_id": "user-1",
                        "filename": "contract.pdf",
                        "content_type": "application/pdf",
                        "object_uri": "s3://documents/contract.pdf",
                    },
                    "publish_attempts": 0,
                }
            )
        if "SET published_at" in sql and self.pause_update:
            self.update_started.set()
            await self.release_update.wait()
        return _MappingResult(None)

    async def commit(self) -> None:
        self.commits += 1


class _SessionContext:
    def __init__(self, session: _Session) -> None:
        self.session = session

    async def __aenter__(self) -> _Session:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


class _SessionFactory:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self) -> _SessionContext:
        return _SessionContext(self.session)


def _relay(session: _Session, publisher: _DirectPublisher) -> OutboxRelay:
    return OutboxRelay(
        database_url="postgresql://unused",
        publisher=publisher,
        session_factory=_SessionFactory(session),
    )


async def test_publish_uses_the_injected_direct_adapter() -> None:
    session = _Session()
    publisher = _DirectPublisher()
    relay = _relay(session, publisher)

    await relay._publish(
        {
            "id": "event-1",
            "event_type": "tasks.documents_ingest",
            "payload": {
                "document_id": "doc-1",
                "user_id": "user-1",
                "filename": "contract.pdf",
                "content_type": "application/pdf",
                "object_uri": "s3://documents/contract.pdf",
            },
        },
        session=session,
    )

    assert publisher.calls == [
        (
            "tasks.documents_ingest",
            {
                "document_id": "doc-1",
                "user_id": "user-1",
                "filename": "contract.pdf",
                "content_type": "application/pdf",
                "object_uri": "s3://documents/contract.pdf",
            },
        )
    ]
    assert session.commits == 1


@pytest.mark.parametrize("receipt", ["", object()])
async def test_publish_requires_a_confirmed_receipt_before_marking_published(
    receipt: object,
) -> None:
    session = _Session()
    publisher = _DirectPublisher(result=receipt)
    relay = _relay(session, publisher)

    await relay._publish(
        {
            "id": "event-1",
            "event_type": "tasks.documents_ingest",
            "payload": {
                "document_id": "doc-1",
                "user_id": "user-1",
                "filename": "contract.pdf",
                "content_type": "application/pdf",
                "object_uri": "s3://documents/contract.pdf",
            },
        },
        session=session,
    )

    assert publisher.calls == [("tasks.documents_ingest", _PAYLOAD)]
    assert not any("SET published_at" in statement for statement in session.statements)
    assert any("publish_attempts" in statement for statement in session.statements)


async def test_startup_scan_publishes_through_the_direct_interface() -> None:
    session = _Session()
    publisher = _DirectPublisher()
    relay = _relay(session, publisher)

    await relay.run_startup_scan()

    assert publisher.calls == [
        (
            "tasks.documents_ingest",
            {
                "document_id": "doc-1",
                "user_id": "user-1",
                "filename": "contract.pdf",
                "content_type": "application/pdf",
                "object_uri": "s3://documents/contract.pdf",
            },
        )
    ]


async def test_dead_letter_stages_idempotent_terminal_event_and_notifies_relay() -> None:
    session = _DeadLetterSession()

    await OutboxRelay._mark_failed("source-event", "broker unavailable", session=session)  # type: ignore[arg-type]

    terminal_insert = next(
        (parameters for sql, parameters in session.calls if "INSERT INTO outbox_events" in sql),
        None,
    )
    assert terminal_insert is not None
    terminal_payload = json.loads(str(terminal_insert["payload"]))
    assert terminal_insert["id"] == terminal_event_id(terminal_payload)
    assert terminal_payload["status"] == "failed"
    assert terminal_payload["task_id"] == "source-event"
    notify_calls = [
        parameters for sql, parameters in session.calls if "pg_notify" in sql
    ]
    assert notify_calls == [{"channel": "outbox_channel", "event_id": terminal_insert["id"]}]
    assert any("dead_letter_events" in sql for sql, _ in session.calls)
    assert session.commits == 1


async def test_publish_refuses_a_payload_the_registry_rejects() -> None:
    session = _Session()
    publisher = _DirectPublisher()
    relay = _relay(session, publisher)

    await relay._publish(
        {
            "id": "event-1",
            "event_type": "tasks.documents_ingest",
            "payload": {"doc_id": "doc-1"},
        },
        session=session,
    )

    assert publisher.calls == []


async def test_drain_waits_for_an_in_flight_publish_and_rejects_new_work() -> None:
    session = _Session(pause_update=True)
    publisher = _DirectPublisher()
    relay = _relay(session, publisher)

    delivery = asyncio.create_task(
        relay._handle_notification(asyncpg_listen.Notification("outbox_channel", "event-1"))
    )
    await session.update_started.wait()

    draining = asyncio.create_task(relay.drain(timeout_seconds=1.0))
    await asyncio.sleep(0)
    assert not draining.done()

    session.release_update.set()
    assert await draining is True
    await delivery

    await relay._handle_notification(asyncpg_listen.Notification("outbox_channel", "event-2"))
    assert len(publisher.calls) == 1


class _ListenerConnection:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _RetryingListenerConnection(_ListenerConnection):
    async def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            msg = "close failed once"
            raise RuntimeError(msg)
        self.closed = True


class _BlockingListenerConnection(_ListenerConnection):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.close_finished = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.release_close.wait()
        self.closed = True
        self.close_finished.set()


async def test_failed_listener_close_retains_ownership_for_retry() -> None:
    relay = _relay(_Session(), _DirectPublisher())
    connection = _RetryingListenerConnection()
    relay._listener_connection = cast("asyncpg.Connection", connection)

    assert await relay._close_listener_connection() is False
    assert relay._listener_connection is connection
    assert await relay._close_listener_connection() is True
    assert connection.closed
    assert connection.close_calls == 2
    assert relay._listener_connection is None


async def test_listener_close_cancellation_keeps_ownership_until_shielded_close_finishes() -> None:
    relay = _relay(_Session(), _DirectPublisher())
    connection = _BlockingListenerConnection()
    relay._listener_connection = cast("asyncpg.Connection", connection)

    closing = asyncio.create_task(relay._close_listener_connection())
    await connection.close_started.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert relay._listener_connection is connection
    connection.release_close.set()
    await asyncio.wait_for(connection.close_finished.wait(), timeout=1.0)
    for _ in range(3):
        await asyncio.sleep(0)
    assert connection.closed
    assert relay._listener_connection is None


async def test_listener_reconnects_and_dispatches_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()
    publisher = _DirectPublisher()
    relay = _relay(session, publisher)
    second_attempt_ready = asyncio.Event()
    release_listener = asyncio.Event()
    attempts = 0
    connections: list[_ListenerConnection] = []

    class _Listener:
        def __init__(self, *, connect: Callable[[], Awaitable[_ListenerConnection]]) -> None:
            self.connect = connect

        async def run(
            self,
            *,
            handler_per_channel: dict[
                str,
                Callable[[asyncpg_listen.Notification | asyncpg_listen.Timeout], Awaitable[None]],
            ],
            **_: object,
        ) -> None:
            nonlocal attempts
            attempts += 1
            connection = await self.connect()
            connections.append(connection)
            if attempts == 1:
                msg = "transient listener failure"
                raise RuntimeError(msg)
            await handler_per_channel["outbox_channel"](
                asyncpg_listen.Notification("outbox_channel", "event-1")
            )
            second_attempt_ready.set()
            await release_listener.wait()

    async def connect(**_: object) -> _ListenerConnection:
        return _ListenerConnection()

    monkeypatch.setattr(relay_module.asyncpg_listen, "NotificationListener", _Listener)
    monkeypatch.setattr(relay_module.asyncpg, "connect", connect)
    monkeypatch.setattr(relay_module, "_LISTENER_BACKOFF_INITIAL_SECONDS", 0.0)
    monkeypatch.setattr(relay_module, "_LISTENER_BACKOFF_JITTER_SECONDS", 0.0)

    listener_task = asyncio.create_task(relay.run_listener())
    await asyncio.wait_for(second_attempt_ready.wait(), timeout=1.0)
    assert attempts == 2
    assert len(publisher.calls) == 1
    assert connections[0].close_calls == 1

    await relay.drain()
    release_listener.set()
    await listener_task
    assert connections[1].close_calls == 1


async def test_listener_backoff_is_interruptible_and_duplicate_run_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = _relay(_Session(), _DirectPublisher())
    second_failure = asyncio.Event()
    attempts = 0

    class _Listener:
        def __init__(self, *, connect: object) -> None:
            _ = connect

        async def run(self, **_: object) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                second_failure.set()
            msg = "listener unavailable"
            raise RuntimeError(msg)

    monkeypatch.setattr(relay_module.asyncpg_listen, "NotificationListener", _Listener)
    monkeypatch.setattr(relay_module, "_LISTENER_BACKOFF_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr(relay_module, "_LISTENER_BACKOFF_MAX_SECONDS", 10.0)
    monkeypatch.setattr(relay_module, "_LISTENER_BACKOFF_JITTER_SECONDS", 0.0)

    listener_task = asyncio.create_task(relay.run_listener())
    await asyncio.sleep(0)
    duplicate_task = asyncio.create_task(relay.run_listener())
    await duplicate_task
    await asyncio.wait_for(second_failure.wait(), timeout=1.0)

    started = time.monotonic()
    await relay.drain(timeout_seconds=1.0)
    await asyncio.wait_for(listener_task, timeout=0.5)
    assert time.monotonic() - started < 0.5
    assert attempts == 2
