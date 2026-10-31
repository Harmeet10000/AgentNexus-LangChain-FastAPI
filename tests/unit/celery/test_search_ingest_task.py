"""Site-search ingest task: stable wire name, retry policy, optional lock.

Mirrors ``test_document_task_retries.py``: the Celery entrypoint raises for
retryable infrastructure failures (the autoretry policy owns retry), runs
without the optional Redis lock, and keeps the typed payload's wire name.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import pytest

from app.features.search.application.ingestion import SearchIngestTerminalError
from app.shared.jobs.names import SEARCH_INGEST
from app.utils import InfrastructureException
from app.worker.jobs.search import record_celery_task_id

if TYPE_CHECKING:
    from typing import Any


def test_search_ingest_keeps_the_stable_wire_name(real_celery: object) -> None:
    del real_celery
    search_tasks = importlib.import_module("app.worker.jobs.search")

    assert search_tasks.search_ingest.name == SEARCH_INGEST


def test_retryable_infrastructure_failures_are_in_the_task_retry_policy(
    real_celery: object,
) -> None:
    del real_celery
    search_tasks = importlib.import_module("app.worker.jobs.search")

    assert InfrastructureException in search_tasks.search_ingest.autoretry_for


def test_terminal_errors_are_exempt_from_autoretry(real_celery: object) -> None:
    """Configuration drift and constraint conflicts must dead-letter, not spin."""
    del real_celery
    search_tasks = importlib.import_module("app.worker.jobs.search")

    assert SearchIngestTerminalError in (search_tasks.search_ingest.dont_autoretry_for or ())


def test_worker_records_celery_task_id_under_the_event_id(real_celery: object) -> None:
    """The relay lets Celery assign the task id; the worker binds it back."""
    del real_celery
    from fakeredis import FakeRedis

    redis = FakeRedis()
    record_celery_task_id(redis, event_id="event-1", task_id="celery-abc")

    assert redis.get("search:task_celery_id:event-1") == b"celery-abc"


def test_celery_id_record_skips_an_incomplete_identity(real_celery: object) -> None:
    """Without both halves there is nothing to bind — and nothing to crash on."""
    del real_celery
    from fakeredis import FakeRedis

    redis = FakeRedis()
    record_celery_task_id(redis, event_id="", task_id="celery-abc")
    record_celery_task_id(redis, event_id="event-2", task_id=None)

    assert redis.get("search:task_celery_id:event-1") is None
    assert redis.get("search:task_celery_id:event-2") is None


def test_task_runs_without_the_optional_redis_lock(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    del real_celery
    search_tasks = importlib.import_module("app.worker.jobs.search")
    task = cast("Any", search_tasks.search_ingest)
    monkeypatch.setattr(search_tasks, "_redis_task_lock_enabled", lambda: False)
    monkeypatch.setattr(
        search_tasks,
        "run_search_ingestion_task",
        _record_success,
    )
    monkeypatch.setattr(
        task,
        "acquire_idempotency_lock",
        lambda *_args, **_kwargs: pytest.fail("Redis lock should not be acquired"),
    )
    monkeypatch.setattr(
        task,
        "mark_idempotency_completed",
        lambda *_args, **_kwargs: pytest.fail("Redis completion should not be written"),
    )

    result = task.run(
        document_id="00000000-0000-0000-0000-0000000000aa",
        tenant_id="tenant-1",
        content="site body",
        content_hash="hash",
        title="Site page",
    )

    assert result == {
        "status": "completed",
        "document_id": "00000000-0000-0000-0000-0000000000aa",
        "tenant_id": "tenant-1",
        "chunk_count": 7,
        "embedding_dim": 768,
    }


async def _record_success(**_kwargs: Any) -> dict[str, object]:
    return {
        "status": "completed",
        "document_id": "00000000-0000-0000-0000-0000000000aa",
        "tenant_id": "tenant-1",
        "chunk_count": 7,
        "embedding_dim": 768,
    }


def test_task_failure_propagates_to_the_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    """Celery entrypoints raise; autoretry owns the retry, the body never swallows."""
    search_tasks = importlib.import_module("app.worker.jobs.search")
    task = cast("Any", search_tasks.search_ingest)
    monkeypatch.setattr(search_tasks, "_redis_task_lock_enabled", lambda: False)

    async def _explode(**_kwargs: Any) -> dict[str, object]:
        message = "database unavailable"
        raise InfrastructureException(message)

    monkeypatch.setattr(search_tasks, "run_search_ingestion_task", _explode)

    with pytest.raises(InfrastructureException):
        task.run(
            document_id="00000000-0000-0000-0000-0000000000aa",
            tenant_id="tenant-1",
            content="site body",
            content_hash="hash",
            title="Site page",
        )
