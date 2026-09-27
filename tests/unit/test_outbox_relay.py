"""Outbox relay collaborator ownership and graceful-drain behavior."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import asyncpg_listen

from app.shared.outbox.relay import OutboxRelay

if TYPE_CHECKING:
    from typing import Any


class _CeleryRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def send_task(self, name: str, *, kwargs: dict[str, object]) -> None:
        self.calls.append((name, kwargs))


class _MappingResult:
    def __init__(self, row: dict[str, object] | None) -> None:
        self._row = row

    def mappings(self) -> _MappingResult:
        return self

    def one_or_none(self) -> dict[str, object] | None:
        return self._row


class _Session:
    def __init__(self, *, pause_update: bool = False) -> None:
        self.pause_update = pause_update
        self.update_started = asyncio.Event()
        self.release_update = asyncio.Event()
        self.commits = 0

    async def execute(self, statement: Any, _parameters: dict[str, object]) -> _MappingResult:
        sql = str(statement)
        if "SELECT id, event_type" in sql:
            return _MappingResult(
                {
                    "id": "event-1",
                    "event_type": "tasks.documents_ingest",
                    "payload": {"doc_id": "doc-1"},
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


def _relay(session: _Session, celery: _CeleryRecorder) -> OutboxRelay:
    return OutboxRelay(
        database_url="postgresql://unused",
        celery_app=celery,
        session_factory=_SessionFactory(session),
    )


async def test_publish_uses_the_injected_celery_application() -> None:
    session = _Session()
    celery = _CeleryRecorder()
    relay = _relay(session, celery)

    await relay._publish(
        {
            "id": "event-1",
            "event_type": "tasks.documents_ingest",
            "payload": {"doc_id": "doc-1"},
        },
        session=session,
    )

    assert celery.calls == [("tasks.documents_ingest", {"doc_id": "doc-1"})]
    assert session.commits == 1


async def test_drain_waits_for_an_in_flight_publish_and_rejects_new_work() -> None:
    session = _Session(pause_update=True)
    celery = _CeleryRecorder()
    relay = _relay(session, celery)

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
    assert len(celery.calls) == 1
