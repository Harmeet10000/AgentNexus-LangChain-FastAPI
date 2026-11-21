"""Queued-ingestion terminal events through the outbox (harden 3.12).

Pins the ingestion status/event contract at its three boundaries:

- worker completion (ready/warnings) stages a terminal outbox event carrying
  the tenant/document/version/attempt/task correlation identity;
- worker failure persistence carries a typed reason and stages the same
  event, while retryable failures stage NOTHING (Celery owns retries);
- the outbox insert is idempotent under redelivery and terminal events are
  never staged from an HTTP module — the request path only records/enqueues.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from returns.result import Success

import app.features.documents.application.commands.upload_document as document_ingest
import app.features.documents.application.ingestion.workflow as ingestion_graph
from app.events.writer import terminal_event_id, with_terminal_outbox
from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.api.http.dto import IngestionJob
from app.features.documents.application.commands.upload_document import (
    process_document_ingestion,
)
from app.features.documents.application.ingestion.chunking.classification import (
    ClassifiedDocument,
    ParsedDocument,
    PreparedChunk,
    QualityWarning,
)
from app.features.documents.application.ingestion.workflow import _record_ingestion_failure

if TYPE_CHECKING:
    from typing import Any

TENANT = "personal:tenant-1"
DOCUMENT_ID = "00000000-0000-0000-0000-000000000001"

_COMPLETE_CHECKPOINT: dict[str, object] = {
    "knowledge_extraction": "complete",
    "knowledge_extraction_classes": [],
    "knowledge_extraction_document_version": 1,
}


# ---------------------------------------------------------------------------
# Fakes (mirroring the truth-fidelity ingestion harness shapes)


class _Session:
    async def commit(self) -> None:
        return None


class _Repo:
    def __init__(self, *, checkpoint_metadata: dict[str, object] | None = None) -> None:
        self.status_calls: list[dict[str, Any]] = []
        self.session = _Session()
        self._checkpoint_metadata = checkpoint_metadata

    async def update_document_status(self, **kwargs: Any) -> Success[None]:
        self.status_calls.append(kwargs)
        return Success(None)

    async def upsert_chunks(self, rows: object) -> Success[None]:
        del rows
        return Success(None)

    async def fetch_status(self, **_kwargs: Any) -> Success[dict[str, Any]]:
        return Success(
            {
                "document_id": DOCUMENT_ID,
                "status": "processing",
                "document_metadata": self._checkpoint_metadata or {},
                "warnings": [],
            }
        )


class _ObjectStore:
    async def get_object(self, *, key: str) -> Success[bytes]:
        del key
        return Success(b"fixture")


def _job() -> IngestionJob:
    return IngestionJob(
        document_id=DOCUMENT_ID,
        user_id="user-1",
        tenant_id=TENANT,
        filename="fixture.md",
        content_type="text/markdown",
        object_uri="s3://bucket/fixture.md",
        task_id="task-42",
    )


def _markdown() -> str:
    return "# Terms\n\nPayment is due in thirty days.\n\nTermination requires notice."


def _chunk(content: str, index: int = 0) -> PreparedChunk:
    return PreparedChunk(chunk_index=index, chunk_kind="generic", content=content)


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    repo: _Repo,
    spy: _TerminalEventSpy,
    *,
    classified: ClassifiedDocument,
) -> None:
    """Stub the model-collaborating stages; the status/outbox writes stay real."""

    async def parse_document(**_kwargs: object) -> ParsedDocument:
        return ParsedDocument(title="Fixture", markdown=_markdown(), page_count=1)

    async def segment_chunks(
        **_kwargs: object,
    ) -> tuple[list[PreparedChunk], list[QualityWarning]]:
        return [_chunk("Payment is due in thirty days.")], []

    async def embed_chunks(**kwargs: object) -> list[dict[str, object]]:
        outcome = cast("Any", kwargs["extraction_outcome"])
        return [
            {
                "id": f"10000000-0000-0000-0000-00000000000{index}",
                "chunk_index": index,
                "custom_metadata": document_ingest._extraction_metadata(outcome),
            }
            for index in range(len(cast("Any", kwargs["chunks"])))
        ]

    monkeypatch.setattr(document_ingest, "parse_document", parse_document)
    monkeypatch.setattr(document_ingest, "classify_document", lambda **_kwargs: classified)
    monkeypatch.setattr(document_ingest, "segment_chunks", segment_chunks)
    monkeypatch.setattr(document_ingest, "_embed_chunks", embed_chunks)
    monkeypatch.setattr(document_ingest, "with_terminal_outbox", spy)


class _TerminalEventSpy:
    """Record every terminal outbox staging with its deterministic id."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, session: object, **kwargs: Any) -> str:
        del session
        payload = cast("dict[str, Any]", kwargs["payload"])
        self.calls.append({"payload": payload, "id": terminal_event_id(payload)})
        return self.calls[-1]["id"]


def _runtime(repo: _Repo) -> Any:
    return SimpleNamespace(
        object_store=_ObjectStore(),
        repo=repo,
        graphiti=None,
        llm=object(),
        extraction=None,
        graph_writer=None,
        idempotency=None,
    )


# ---------------------------------------------------------------------------
# Worker completion → ready/warnings terminal event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_completion_stages_a_ready_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _TerminalEventSpy()
    repo = _Repo(checkpoint_metadata=_COMPLETE_CHECKPOINT)
    _patch_pipeline(monkeypatch, repo, spy, classified=ClassifiedDocument(document_kind="generic"))

    result = await process_document_ingestion(job=_job(), runtime=_runtime(repo))

    assert isinstance(result, Success)
    final_status = repo.status_calls[-1]["status"]
    assert final_status in {"completed", "completed_with_warnings"}
    assert len(spy.calls) == 1, "exactly one terminal event for one completion"
    payload = spy.calls[0]["payload"]
    assert payload["status"] == final_status
    # Correlation identity: the UI notification is bound to the worker attempt.
    assert payload["document_id"] == DOCUMENT_ID
    assert payload["tenant_id"] == TENANT
    assert payload["document_version"] == 1
    assert payload["attempt"] == 0
    assert payload["task_id"] == "task-42"


@pytest.mark.asyncio
async def test_worker_completion_with_warnings_stages_a_ready_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _TerminalEventSpy()
    repo = _Repo(checkpoint_metadata=_COMPLETE_CHECKPOINT)
    _patch_pipeline(
        monkeypatch,
        repo,
        spy,
        classified=ClassifiedDocument(
            document_kind="generic",
            warnings=[
                QualityWarning(
                    stage="classification",
                    code="LOW_CONFIDENCE_KIND",
                    message="Kind inferred with low confidence",
                    severity="warning",
                )
            ],
        ),
    )

    result = await process_document_ingestion(job=_job(), runtime=_runtime(repo))

    assert isinstance(result, Success)
    assert repo.status_calls[-1]["status"] == "completed_with_warnings"
    assert len(spy.calls) == 1
    assert spy.calls[0]["payload"]["status"] == "completed_with_warnings"


# ---------------------------------------------------------------------------
# Worker failure persistence → typed reason + terminal event
# ---------------------------------------------------------------------------


def _failure_repo() -> tuple[MagicMock, AsyncMock]:
    repo = MagicMock(spec=DocumentRepository)
    repo.update_document_status = AsyncMock(return_value=Success(None))
    repo.session = _Session()
    return repo, repo.update_document_status


@pytest.mark.asyncio
async def test_non_retryable_failure_persists_typed_reason_and_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, status_updater = _failure_repo()
    spy = _TerminalEventSpy()
    monkeypatch.setattr(ingestion_graph, "with_terminal_outbox", spy)

    await _record_ingestion_failure(
        repo,  # type: ignore[arg-type]
        document_id=DOCUMENT_ID,
        user_id="user-1",
        tenant_id=TENANT,
        document_version=1,
        attempt=3,
        task_id="task-42",
        error_code="INGESTION_RETRY_EXHAUSTED",
        error_message="retry budget exhausted",
        retryable=False,
    )

    status_updater.assert_awaited_once()
    write = status_updater.await_args.kwargs
    assert write["status"] == "failed"
    assert write["metadata_"]["ingestion_error"]["code"] == "INGESTION_RETRY_EXHAUSTED"
    assert len(spy.calls) == 1
    payload = spy.calls[0]["payload"]
    assert payload["status"] == "failed"
    assert payload["error_code"] == "INGESTION_RETRY_EXHAUSTED"
    assert payload["attempt"] == 3
    assert payload["task_id"] == "task-42"
    assert payload["tenant_id"] == TENANT
    assert payload["document_version"] == 1


@pytest.mark.asyncio
async def test_retryable_failure_persists_nothing_worker_owns_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, status_updater = _failure_repo()
    spy = _TerminalEventSpy()
    monkeypatch.setattr(ingestion_graph, "with_terminal_outbox", spy)

    await _record_ingestion_failure(
        repo,  # type: ignore[arg-type]
        document_id=DOCUMENT_ID,
        tenant_id=TENANT,
        error_code="PROVIDER_TRANSIENT",
        error_message="provider unavailable",
        retryable=True,
    )

    status_updater.assert_not_awaited()
    assert spy.calls == [], "a retryable failure must never stage a terminal event"


# ---------------------------------------------------------------------------
# Outbox insert idempotency
# ---------------------------------------------------------------------------


class _OutboxSession:
    """Scripted session: the insert may conflict (redelivery), notify may not run."""

    def __init__(self, *, insert_returns: list[str | None]) -> None:
        self.statements: list[str] = []
        self._insert_returns = iter(insert_returns)

    async def execute(self, statement: Any, _parameters: Any = None) -> Any:
        sql = str(statement)
        self.statements.append(sql)
        returned = next(self._insert_returns, None)

        class _Scalar:
            def scalar_one_or_none(self) -> str | None:
                return returned

        return _Scalar()


@pytest.mark.asyncio
async def test_duplicate_terminal_event_inserts_once_and_notifies_once() -> None:
    payload: dict[str, object] = {
        "document_id": DOCUMENT_ID,
        "user_id": "user-1",
        "tenant_id": TENANT,
        "status": "completed",
        "document_version": 1,
        "attempt": 0,
        "task_id": "task-42",
    }
    first = _OutboxSession(insert_returns=["event-id"])
    second = _OutboxSession(insert_returns=[None])  # ON CONFLICT DO NOTHING

    event_id_1 = await with_terminal_outbox(
        first,  # type: ignore[arg-type]
        document_id=DOCUMENT_ID,
        payload=dict(payload),
    )
    event_id_2 = await with_terminal_outbox(
        second,  # type: ignore[arg-type]
        document_id=DOCUMENT_ID,
        payload=dict(payload),
    )

    assert event_id_1 == event_id_2 == terminal_event_id(payload)
    assert any("pg_notify" in sql for sql in first.statements), "first insert notifies"
    assert not any("pg_notify" in sql for sql in second.statements), "conflict never notifies"
    insert_sqls = [sql for sql in first.statements if "INSERT INTO outbox_events" in sql]
    assert len(insert_sqls) == 1


# ---------------------------------------------------------------------------
# Worker-vs-HTTP ownership
# ---------------------------------------------------------------------------


def test_terminal_events_are_never_staged_from_an_http_module() -> None:
    """The request path only records/enqueues; terminal events are staged by
    the worker boundary, the relay dead-letter bridge, or the outbox writer."""
    src_root = Path("src/app")
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        parts = {part.lower() for part in path.parts}
        if "api" not in parts:
            continue
        if "with_terminal_outbox" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert offenders == []
