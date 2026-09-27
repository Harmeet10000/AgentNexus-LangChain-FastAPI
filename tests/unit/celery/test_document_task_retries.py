"""Document ingestion task retry and optional-lock contracts."""

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from typing import Any


def test_retryable_infrastructure_failures_are_in_the_task_retry_policy(
    real_celery: object,
) -> None:
    del real_celery
    document_tasks = importlib.import_module("tasks.document_tasks")
    infrastructure_exception = importlib.import_module("app.utils").InfrastructureException

    assert infrastructure_exception in document_tasks.ingest_document.autoretry_for


def test_document_ingestion_runs_without_the_optional_redis_lock(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    del real_celery
    document_tasks = importlib.import_module("tasks.document_tasks")
    task = cast("Any", document_tasks.ingest_document)
    resources = SimpleNamespace(engine=object(), ingestion_graph=object(), session_local=object())
    monkeypatch.setattr(document_tasks, "_redis_task_lock_enabled", lambda: False)
    monkeypatch.setattr(document_tasks, "get_document_worker_resources", lambda: resources)
    monkeypatch.setattr(
        document_tasks,
        "run_on_document_worker_loop",
        lambda _operation: {"status": "completed", "document_id": "doc-1"},
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
        document_id="doc-1",
        user_id="user-1",
        filename="contract.pdf",
        content_type="application/pdf",
        object_uri="s3://documents/contract.pdf",
    )

    assert result == {"status": "completed", "document_id": "doc-1"}
