"""Site-search ingest task: stable wire name, retry policy, optional lock.

Mirrors ``test_document_task_retries.py``: the Celery entrypoint raises for
retryable infrastructure failures (the autoretry policy owns retry), runs
without the optional Redis lock, and keeps the typed payload's wire name.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from app.features.search.application.ingestion import SearchIngestTerminalError
from app.features.search.application.service import bump_tenant_cache_version
from app.shared.jobs.names import SEARCH_INGEST
from app.utils import InfrastructureException
from app.worker.jobs.search import record_celery_task_id

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture(autouse=True)
def _fresh_worker_loop():
    """Each test gets a fresh persistent loop matching its fresh FakeRedis.

    The worker runner retains one loop per process; other test modules close
    it, and a fakeredis connection pooled on a closed loop raises
    "Event loop is closed" in later tests.
    """
    from app.worker.execution import runtime

    runtime.close()
    yield
    runtime.close()


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


@pytest.mark.asyncio
async def test_worker_records_celery_task_id_under_the_event_id(real_celery: object) -> None:
    """The relay lets Celery assign the task id; the worker binds it back.

    The worker's client is redis.asyncio: the record MUST be awaited, or the
    set coroutine is discarded and the mapping never lands (the exact defect
    the round-3 review flagged)."""
    del real_celery
    from fakeredis.aioredis import FakeRedis

    redis = FakeRedis()
    await record_celery_task_id(redis, event_id="event-1", task_id="celery-abc")

    assert await redis.get("search:task_celery_id:event-1") == b"celery-abc"


@pytest.mark.asyncio
async def test_celery_id_record_skips_an_incomplete_identity(real_celery: object) -> None:
    """Without both halves there is nothing to bind — and nothing to crash on."""
    del real_celery
    from fakeredis.aioredis import FakeRedis

    redis = FakeRedis()
    await record_celery_task_id(redis, event_id="", task_id="celery-abc")
    await record_celery_task_id(redis, event_id="event-2", task_id=None)

    assert await redis.get("search:task_celery_id:event-1") is None
    assert await redis.get("search:task_celery_id:event-2") is None


def test_successful_run_records_id_and_bumps_cache_version(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    """Sync test: task.run() owns the event loop via asyncio.run."""
    del real_celery
    from fakeredis.aioredis import FakeRedis

    search_tasks = importlib.import_module("app.worker.jobs.search")
    task = cast("Any", search_tasks.search_ingest)
    redis = FakeRedis()
    monkeypatch.setattr(search_tasks, "_redis_task_lock_enabled", lambda: True)
    monkeypatch.setattr(
        search_tasks,
        "get_settings",
        lambda: SimpleNamespace(REDIS_URL="redis://", SEARCH_TASK_OWNER_TTL_SECONDS=86_400),
    )
    monkeypatch.setattr(task, "acquire_idempotency_lock", lambda *_a, **_k: True)
    monkeypatch.setattr(task, "mark_idempotency_completed", lambda *_a, **_k: None)
    monkeypatch.setattr(task, "get_redis_client", lambda: redis)
    # The resolved task's `request` is read-only; spy on the record seam and
    # hand it the dispatch id a real request would carry.
    recorded: list[tuple[str | None, str | None]] = []
    real_record = search_tasks.record_celery_task_id

    async def _recording_spy(
        redis_client: Any, *, event_id: str | None, task_id: str | None
    ) -> None:
        task_id = task_id or "celery-abc"
        recorded.append((event_id, task_id))
        await real_record(redis_client, event_id=event_id, task_id=task_id)

    monkeypatch.setattr(search_tasks, "record_celery_task_id", _recording_spy)
    bumped: list[str] = []
    real_bump = search_tasks.bump_tenant_cache_version

    async def _bump_spy(redis_client: Any, *, tenant_id: str) -> None:
        bumped.append(tenant_id)
        await real_bump(redis_client, tenant_id=tenant_id)

    monkeypatch.setattr(search_tasks, "bump_tenant_cache_version", _bump_spy)

    async def _success(**_kwargs: Any) -> dict[str, object]:
        return {"status": "completed", "chunk_count": 3}

    monkeypatch.setattr(search_tasks, "run_search_ingestion_task", _success)

    result = task.run(
        document_id="00000000-0000-0000-0000-0000000000aa",
        tenant_id="tenant-1",
        content="body",
        content_hash="hash",
        title="Site page",
        event_id="event-1",
    )

    assert result["status"] == "completed"
    # The worker bound its dispatch id to the event id the client polls.
    assert recorded == [("event-1", "celery-abc")]
    # The completion path invalidated the tenant's cached searches (the bump's
    # own digest behavior is pinned in the service cache-version test).
    assert bumped == ["tenant-1"]


def test_lock_disabled_run_never_constructs_a_redis_client(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    """An empty REDIS_URL gates everything cache-shaped BEFORE get_redis_client."""
    del real_celery
    search_tasks = importlib.import_module("app.worker.jobs.search")
    task = cast("Any", search_tasks.search_ingest)
    monkeypatch.setattr(search_tasks, "get_settings", lambda: SimpleNamespace(REDIS_URL=""))

    def _no_client(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Redis client constructed despite empty REDIS_URL")

    resolved_task = task._get_current_object()
    monkeypatch.setattr(resolved_task, "get_redis_client", _no_client)

    async def _success(**_kwargs: Any) -> dict[str, object]:
        return {"status": "completed", "chunk_count": 0}

    monkeypatch.setattr(search_tasks, "run_search_ingestion_task", _success)

    result = task.run(
        document_id="00000000-0000-0000-0000-0000000000aa",
        tenant_id="tenant-1",
        content="body",
        content_hash="hash",
        title="Site page",
    )

    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_cache_bump_is_best_effort(real_celery: object) -> None:
    """A dead cache must not fail the completed ingest (degradation boundary)."""
    del real_celery

    class _DeadRedis:
        async def incr(self, *_args: Any) -> Any:
            message = "cache down"
            raise RuntimeError(message)

    # No exception escapes: the bump failure is logged and swallowed.
    await bump_tenant_cache_version(_DeadRedis(), tenant_id="tenant-1")


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
