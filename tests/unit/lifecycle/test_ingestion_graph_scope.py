"""The compiled ingestion graph keeps job-scoped objects out of graph state."""

from __future__ import annotations

import inspect
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from returns.result import Success
from sqlalchemy.ext.asyncio import AsyncSession

import app.features.documents.ingestion_graph as ingestion_graph_module
from app.utils.storage import StorageService
from app.features.documents.ingestion_graph import (
    DocumentIngestionState,
    _make_ingest_document_node,
    build_document_ingestion_graph,
)
from app.features.documents.repository import DocumentRepository

pytestmark = pytest.mark.unit


def test_compiled_graph_captures_no_repository_or_session() -> None:
    object_store = MagicMock()
    graphiti = MagicMock()
    llm = MagicMock()
    ingest = AsyncMock()

    node = _make_ingest_document_node(
        object_store=object_store,
        graphiti=graphiti,
        ingest_document_fn=ingest,
        llm=llm,
    )
    graph = build_document_ingestion_graph(
        object_store=object_store,
        graphiti=graphiti,
        ingest_document_fn=ingest,
        llm=llm,
    )

    captured = inspect.getclosurevars(node).nonlocals.values()
    assert not any(isinstance(value, (DocumentRepository, AsyncSession)) for value in captured)
    assert "repo" not in get_type_hints(DocumentIngestionState)
    assert "repository" not in get_type_hints(DocumentIngestionState)
    assert "session" not in get_type_hints(DocumentIngestionState)
    assert graph is not None


@pytest.mark.asyncio
async def test_ingestion_node_prefers_job_scoped_idempotency() -> None:
    ingest = AsyncMock(return_value=Success({"status": "completed"}))
    default_idempotency = object()
    job_idempotency = object()
    repo = MagicMock(spec=DocumentRepository)
    repo.session = MagicMock(spec=AsyncSession)
    node = _make_ingest_document_node(
        object_store=MagicMock(spec=StorageService),
        graphiti=None,
        ingest_document_fn=ingest,
        llm=FakeListChatModel(responses=["unused"]),
        idempotency=default_idempotency,
    )

    await node(
        {
            "document_id": "doc-1",
            "user_id": "user-1",
            "filename": "fixture.txt",
            "content_type": "text/plain",
            "object_uri": "s3://bucket/fixture.txt",
        },
        {
            "configurable": {
                "document_repository": repo,
                "document_idempotency": job_idempotency,
            }
        },  # type: ignore[arg-type]
    )

    runtime = ingest.await_args.kwargs["runtime"]
    assert runtime.idempotency is job_idempotency
    assert runtime.transaction_checkpoint == repo.session.commit


@pytest.mark.asyncio
async def test_retryable_worker_failure_is_not_persisted_as_terminal_status() -> None:
    async def fail_before_processing(**_: object) -> object:
        error_message = "parser unavailable"
        raise RuntimeError(error_message)

    repo = MagicMock(spec=DocumentRepository)
    repo.session = MagicMock(spec=AsyncSession)
    repo.update_document_status = AsyncMock(return_value=Success(None))
    node = _make_ingest_document_node(
        object_store=MagicMock(spec=StorageService),
        graphiti=None,
        ingest_document_fn=fail_before_processing,
        llm=FakeListChatModel(responses=["unused"]),
    )

    result = await node(
        {
            "document_id": "doc-1",
            "user_id": "user-1",
            "filename": "fixture.txt",
            "content_type": "text/plain",
            "object_uri": "s3://bucket/fixture.txt",
        },
        {"configurable": {"document_repository": repo}},  # type: ignore[arg-type]
    )

    assert result["status"] == "failed"
    repo.update_document_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_retryable_failure_does_not_stage_terminal_outbox_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_outbox = AsyncMock()
    monkeypatch.setattr(ingestion_graph_module, "with_terminal_outbox", terminal_outbox)
    repo = MagicMock(spec=DocumentRepository)
    repo.session = MagicMock(spec=AsyncSession)
    repo.update_document_status = AsyncMock(return_value=Success(None))

    async def fail_before_processing(**_: object) -> object:
        message = "parser unavailable"
        raise RuntimeError(message)

    node = _make_ingest_document_node(
        object_store=MagicMock(spec=StorageService),
        graphiti=None,
        ingest_document_fn=fail_before_processing,
        llm=FakeListChatModel(responses=["unused"]),
    )

    result = await node(
        {
            "document_id": "doc-1",
            "user_id": "user-1",
            "tenant_id": "tenant-1",
            "filename": "fixture.txt",
            "content_type": "text/plain",
            "object_uri": "s3://bucket/fixture.txt",
        },
        {"configurable": {"document_repository": repo}},  # type: ignore[arg-type]
    )

    assert result["status"] == "failed"
    repo.update_document_status.assert_not_awaited()
    terminal_outbox.assert_not_awaited()


@pytest.mark.asyncio
async def test_retryable_failure_can_be_replayed_to_success_without_failed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_outbox = AsyncMock()
    monkeypatch.setattr(ingestion_graph_module, "with_terminal_outbox", terminal_outbox)
    repo = MagicMock(spec=DocumentRepository)
    repo.session = MagicMock(spec=AsyncSession)
    transient = RuntimeError("parser unavailable")
    ingest = AsyncMock(
        side_effect=[transient, Success({"status": "completed", "document_id": "doc-1"})]
    )
    node = _make_ingest_document_node(
        object_store=MagicMock(spec=StorageService),
        graphiti=None,
        ingest_document_fn=ingest,
        llm=FakeListChatModel(responses=["unused"]),
    )
    state = {
        "document_id": "doc-1",
        "user_id": "user-1",
        "tenant_id": "tenant-1",
        "document_version": 2,
        "attempt": 0,
        "filename": "fixture.txt",
        "content_type": "text/plain",
        "object_uri": "s3://bucket/fixture.txt",
    }
    config = {"configurable": {"document_repository": repo}}  # type: ignore[arg-type]

    first = await node(state, config)
    second = await node({**state, "attempt": 1}, config)

    assert first["status"] == "failed"
    assert first["error_retryable"] is True
    assert second == {"status": "completed", "document_id": "doc-1"}
    repo.update_document_status.assert_not_awaited()
    terminal_outbox.assert_not_awaited()
