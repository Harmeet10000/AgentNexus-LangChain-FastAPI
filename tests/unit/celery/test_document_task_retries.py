"""Document ingestion task retry and optional-lock contracts."""

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from typing import Any


def test_retryable_infrastructure_failures_are_in_the_task_retry_policy(
    real_celery: object,
) -> None:
    del real_celery
    document_tasks = importlib.import_module("app.worker.jobs.documents")
    infrastructure_exception = importlib.import_module("app.utils").InfrastructureException

    assert infrastructure_exception in document_tasks.ingest_document.autoretry_for


def test_document_ingestion_runs_without_the_optional_redis_lock(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    del real_celery
    document_tasks = importlib.import_module("app.worker.jobs.documents")
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


def test_advisory_lock_skip_releases_redis_without_marking_completion(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    del real_celery
    document_tasks = importlib.import_module("app.worker.jobs.documents")
    task = cast("Any", document_tasks.ingest_document)
    resources = SimpleNamespace(engine=object(), ingestion_graph=object(), session_local=object())
    release = MagicMock()
    complete = MagicMock()
    monkeypatch.setattr(document_tasks, "_redis_task_lock_enabled", lambda: True)
    monkeypatch.setattr(document_tasks, "get_document_worker_resources", lambda: resources)
    monkeypatch.setattr(
        document_tasks,
        "run_on_document_worker_loop",
        lambda _operation: {"status": "skipped", "document_id": "doc-1"},
    )
    monkeypatch.setattr(task, "acquire_idempotency_lock", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(task, "release_idempotency_processing_lock", release)
    monkeypatch.setattr(task, "mark_idempotency_completed", complete)

    result = task.run(
        document_id="doc-1",
        user_id="user-1",
        filename="contract.pdf",
        content_type="application/pdf",
        object_uri="s3://documents/contract.pdf",
    )

    assert result == {"status": "skipped", "document_id": "doc-1"}
    release.assert_called_once_with("documents-ingest:personal:user-1:doc-1:v1")
    complete.assert_not_called()


def test_document_lock_key_is_scoped_to_tenant_document_and_version(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    del real_celery
    document_tasks = importlib.import_module("app.worker.jobs.documents")
    task = cast("Any", document_tasks.ingest_document)
    resources = SimpleNamespace(engine=object(), ingestion_graph=object(), session_local=object())
    acquired: list[str] = []
    completed: list[str] = []
    monkeypatch.setattr(document_tasks, "_redis_task_lock_enabled", lambda: True)
    monkeypatch.setattr(document_tasks, "get_document_worker_resources", lambda: resources)
    monkeypatch.setattr(
        document_tasks,
        "run_on_document_worker_loop",
        lambda _operation: {"status": "completed", "document_id": "doc-1"},
    )
    monkeypatch.setattr(
        task,
        "acquire_idempotency_lock",
        lambda key, **_kwargs: acquired.append(key) or True,
    )
    monkeypatch.setattr(
        task,
        "mark_idempotency_completed",
        lambda key, **_kwargs: completed.append(key),
    )

    result = task.run(
        document_id="doc-1",
        user_id="user-1",
        tenant_id="tenant-a",
        document_version=7,
        filename="contract.pdf",
        content_type="application/pdf",
        object_uri="s3://documents/contract.pdf",
    )

    expected = "documents-ingest:tenant-a:doc-1:v7"
    assert result == {"status": "completed", "document_id": "doc-1"}
    assert acquired == [expected]
    assert completed == [expected]


def test_retryable_failure_releases_processing_lock_without_completion_marker(
    monkeypatch: pytest.MonkeyPatch,
    real_celery: object,
) -> None:
    del real_celery
    document_tasks = importlib.import_module("app.worker.jobs.documents")
    task = cast("Any", document_tasks.ingest_document)
    infrastructure_exception = importlib.import_module("app.utils").InfrastructureException
    resources = SimpleNamespace(engine=object(), ingestion_graph=object(), session_local=object())
    release = MagicMock()
    complete = MagicMock()
    monkeypatch.setattr(document_tasks, "_redis_task_lock_enabled", lambda: True)
    monkeypatch.setattr(document_tasks, "get_document_worker_resources", lambda: resources)
    monkeypatch.setattr(
        document_tasks,
        "run_on_document_worker_loop",
        lambda _operation: {
            "status": "failed",
            "error_retryable": True,
            "error_code": "INGESTION_CHECKPOINT_FAILED",
            "error_message": "database unavailable",
        },
    )
    monkeypatch.setattr(task, "acquire_idempotency_lock", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(task, "release_idempotency_processing_lock", release)
    monkeypatch.setattr(task, "mark_idempotency_completed", complete)

    with pytest.raises(infrastructure_exception, match="database unavailable"):
        task.run(
            document_id="doc-1",
            user_id="user-1",
            tenant_id="tenant-a",
            document_version=7,
            filename="contract.pdf",
            content_type="application/pdf",
            object_uri="s3://documents/contract.pdf",
        )

    expected = "documents-ingest:tenant-a:doc-1:v7"
    release.assert_called_once_with(expected)
    complete.assert_not_called()
