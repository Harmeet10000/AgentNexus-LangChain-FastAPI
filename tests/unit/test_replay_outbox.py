"""Public replay tests for the direct JobBus publication seam."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from celery.exceptions import CeleryError
from pydantic import ValidationError

import app.worker.operations.replay_outbox as replay_module

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

    from app.shared.jobs import EnqueueOptions, TypedJob


_PAYLOAD = {
    "document_id": "doc-1",
    "user_id": "user-1",
    "filename": "contract.pdf",
    "content_type": "application/pdf",
    "object_uri": "s3://documents/contract.pdf",
}


class _Publisher:
    def __init__(self, *, failure: Exception | None = None, result: object = "task-1") -> None:
        self.failure = failure
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def publish(self, job: TypedJob[Any], *, options: EnqueueOptions) -> object:
        _ = options
        if self.failure is not None:
            raise self.failure
        self.calls.append((job.task_name, job.wire_payload()))
        return self.result


class _Engine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


class _Session:
    def __init__(self) -> None:
        self.deleted: list[object] = []

    async def execute(self, *, statement: object) -> None:
        self.deleted.append(statement)


class _BeginContext:
    def __init__(self, session: _Session) -> None:
        self.session = session

    async def __aenter__(self) -> _Session:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


class _SessionFactory:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def begin(self) -> _BeginContext:
        return _BeginContext(self.session)


def _patch_replay_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, object]],
) -> tuple[_Engine, _Session]:
    engine = _Engine()
    session = _Session()
    factory = _SessionFactory(session)

    async def fetch(_session_factory: object, *, limit: int) -> AsyncIterator[dict[str, object]]:
        for event in events[:limit]:
            yield event

    monkeypatch.setattr(replay_module, "get_database_url", lambda: "postgresql://unused")
    monkeypatch.setattr(replay_module, "create_async_engine", lambda _url: engine)
    monkeypatch.setattr(replay_module, "async_sessionmaker", lambda _engine: factory)
    monkeypatch.setattr(replay_module, "_fetch_dead_letters", fetch)
    return engine, session


@pytest.mark.asyncio
async def test_replay_publishes_typed_job_then_deletes_confirmed_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "original_event_id": "event-1",
            "event_type": "tasks.documents_ingest",
            "payload": _PAYLOAD,
        }
    ]
    engine, session = _patch_replay_dependencies(monkeypatch, events)
    publisher = _Publisher()

    assert await replay_module.replay(limit=10, publisher=publisher) == 1
    assert publisher.calls == [("tasks.documents_ingest", _PAYLOAD)]
    assert len(session.deleted) == 1
    assert engine.disposed


@pytest.mark.asyncio
async def test_replay_keeps_invalid_and_failed_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "original_event_id": "invalid-event",
            "event_type": "tasks.not_in_catalog",
            "payload": {},
        },
        {
            "original_event_id": "failed-event",
            "event_type": "tasks.documents_ingest",
            "payload": _PAYLOAD,
        },
    ]
    _engine, session = _patch_replay_dependencies(monkeypatch, events)
    publisher = _Publisher(failure=CeleryError("broker unavailable"))

    assert await replay_module.replay(limit=10, publisher=publisher) == 0
    assert publisher.calls == []
    assert session.deleted == []


@pytest.mark.asyncio
async def test_replay_discards_raw_validation_failure_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "original_event_id": "validation-event",
            "event_type": "tasks.documents_ingest",
            "payload": _PAYLOAD,
        }
    ]
    _engine, session = _patch_replay_dependencies(monkeypatch, events)
    failure = ValidationError.from_exception_data(
        "Payload",
        [{"type": "missing", "loc": ("document_id",), "input": {}}],
    )

    assert await replay_module.replay(limit=10, publisher=_Publisher(failure=failure)) == 0
    assert session.deleted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["", object()])
async def test_replay_keeps_event_when_adapter_returns_invalid_receipt(
    monkeypatch: pytest.MonkeyPatch,
    result: object,
) -> None:
    events = [
        {
            "original_event_id": "empty-receipt-event",
            "event_type": "tasks.documents_ingest",
            "payload": _PAYLOAD,
        }
    ]
    _engine, session = _patch_replay_dependencies(monkeypatch, events)
    publisher = _Publisher(result=result)

    assert await replay_module.replay(limit=10, publisher=publisher) == 0
    assert publisher.calls == [("tasks.documents_ingest", _PAYLOAD)]
    assert session.deleted == []
