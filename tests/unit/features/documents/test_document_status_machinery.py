"""Typed document status machinery (harden 3.11).

Pins the queued-ingestion lifecycle contract:

- the typed ``DocumentStatus`` mapping — no untyped status can be persisted;
- the extraction-reason guard — an ``extraction_incomplete=True`` write always
  carries a typed failure reason (the live run found 12/15 documents failed
  with no reason stored and no worker-log trace);
- stuck-age detection — a non-terminal status exposes its age, a terminal one
  never reads as stuck;
- the dead-letter-to-failed transition — an exhausted dispatch never leaves a
  document in ``received`` and never clobbers a completed document.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from returns.result import Failure, Success

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.domain.document import (
    DOCUMENT_STATUS_VALUES,
    EXTRACTION_FAILURE_UNRECORDED,
    TERMINAL_DOCUMENT_STATUSES,
    DocumentStatus,
)
from app.features.documents.domain.errors import DocumentValidationError

if TYPE_CHECKING:
    from typing import Any

TENANT = "tenant-1"
DOCUMENT_ID = "3f6d3b1e-0000-4000-8000-000000000001"


class _Result:
    """Mimic the one SQLAlchemy result surface the repository uses."""

    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row

    def mappings(self) -> _Result:
        return self

    def one_or_none(self) -> dict[str, Any] | None:
        return self._row


class _ScriptedSession:
    """Record every statement; answer selects with scripted rows."""

    def __init__(self, rows: list[dict[str, Any] | None]) -> None:
        self.statements: list[Any] = []
        self._rows = iter(rows)

    async def execute(self, statement: Any) -> _Result:
        self.statements.append(statement)
        return _Result(next(self._rows, None))

    async def rollback(self) -> None:
        return None


def _document_row(*, status: str, ingestion_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "tenant_id": TENANT,
        "status": status,
        "metadata_": {"ingestion_state": ingestion_state} if ingestion_state else {},
    }


# ---------------------------------------------------------------------------
# Typed status mapping/constraint
# ---------------------------------------------------------------------------


def test_the_status_mapping_covers_the_known_lifecycle() -> None:
    assert {
        "received",
        "processing",
        "parsed",
        "stored_postgres",
        "completed",
        "completed_with_warnings",
        "failed",
        # STRIDE D-3: the embedding provider's daily quota was exhausted. A
        # parked document, deliberately NOT terminal — the next quota window's
        # idempotent resume selects non-terminal rows.
        "quota_exhausted",
    } == DOCUMENT_STATUS_VALUES
    assert {
        DocumentStatus.COMPLETED.value,
        DocumentStatus.COMPLETED_WITH_WARNINGS.value,
        DocumentStatus.FAILED.value,
    } == TERMINAL_DOCUMENT_STATUSES


@pytest.mark.asyncio
async def test_unknown_status_is_refused_by_the_status_updater() -> None:
    session = _ScriptedSession([])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.update_document_status(
        document_id=DOCUMENT_ID,
        status="completed-ish",
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), DocumentValidationError)
    assert session.statements == [], "an untyped status must reach no SQL"


@pytest.mark.asyncio
async def test_unknown_status_is_refused_by_document_creation() -> None:
    session = _ScriptedSession([])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.create_document(
        user_id="user-1",
        tenant_id=TENANT,
        title="t",
        source_uri=None,
        object_uri="s3://bucket/key",
        content_hash="a" * 64,
        document_kind="generic",
        status="queued_forever",
        jurisdiction=None,
        contract_type=None,
        parties=[],
        metadata_={},
    )

    assert isinstance(result, Failure)
    assert isinstance(result.failure(), DocumentValidationError)
    assert session.statements == [], "an untyped status must reach no SQL"


# ---------------------------------------------------------------------------
# Extraction-reason guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extraction_incomplete_without_a_reason_persists_the_typed_marker() -> None:
    session = _ScriptedSession([None])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.update_document_status(
        document_id=DOCUMENT_ID,
        status="failed",
        metadata_={"ingestion_state": {"attempt": 1}},
        extraction_incomplete=True,
    )

    assert isinstance(result, Success)
    params = session.statements[0].compile().params
    payload = json.loads(params["metadata_"])
    assert payload["extraction_failure_reason"] == EXTRACTION_FAILURE_UNRECORDED
    assert payload["ingestion_state"] == {"attempt": 1}, "caller metadata is preserved"


@pytest.mark.asyncio
async def test_extraction_incomplete_with_a_typed_reason_is_left_untouched() -> None:
    session = _ScriptedSession([None])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.update_document_status(
        document_id=DOCUMENT_ID,
        status="failed",
        metadata_={"knowledge_extraction_failure": "BUDGET_EXHAUSTED"},
        extraction_incomplete=True,
    )

    assert isinstance(result, Success)
    payload = json.loads(session.statements[0].compile().params["metadata_"])
    assert payload["knowledge_extraction_failure"] == "BUDGET_EXHAUSTED"
    assert "extraction_failure_reason" not in payload


@pytest.mark.asyncio
async def test_complete_extraction_write_carries_no_failure_marker() -> None:
    session = _ScriptedSession([None])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.update_document_status(
        document_id=DOCUMENT_ID,
        status="parsed",
        metadata_={"knowledge_extraction": "complete"},
        extraction_incomplete=False,
    )

    assert isinstance(result, Success)
    payload = json.loads(session.statements[0].compile().params["metadata_"])
    assert "extraction_failure_reason" not in payload


# ---------------------------------------------------------------------------
# Stuck-age detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_terminal_status_exposes_its_age() -> None:
    stuck_updated_at = datetime.now(tz=UTC) - timedelta(hours=3)
    session = _ScriptedSession([{"status": "processing", "updated_at": stuck_updated_at}])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.ingestion_status_age(document_id=DOCUMENT_ID, tenant_id=TENANT)

    assert isinstance(result, Success)
    observation = result.unwrap()
    assert observation is not None
    status, age_seconds = observation
    assert status == "processing"
    assert age_seconds >= 2 * 3600, "a 3-hour-old processing row reads as stuck"


@pytest.mark.asyncio
async def test_terminal_status_is_never_stuck() -> None:
    for terminal in sorted(TERMINAL_DOCUMENT_STATUSES):
        session = _ScriptedSession(
            [{"status": terminal, "updated_at": datetime.now(tz=UTC) - timedelta(days=30)}]
        )
        repository = DocumentRepository(session)  # type: ignore[arg-type]

        result = await repository.ingestion_status_age(document_id=DOCUMENT_ID, tenant_id=TENANT)

        assert isinstance(result, Success)
        assert result.unwrap() is None, f"{terminal} must never read as stuck"


@pytest.mark.asyncio
async def test_missing_or_foreign_document_reads_as_not_stuck() -> None:
    session = _ScriptedSession([None])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.ingestion_status_age(document_id=DOCUMENT_ID, tenant_id=TENANT)

    assert isinstance(result, Success)
    assert result.unwrap() is None


# ---------------------------------------------------------------------------
# Dead-letter-to-failed transition
# ---------------------------------------------------------------------------


_IDENTITY = {
    "tenant_id": TENANT,
    "document_version": 1,
    "attempt": 0,
    "task_id": "task-1",
}
_FAILURE_METADATA: dict[str, object] = {
    "ingestion_error": {
        "code": "INGESTION_RETRY_EXHAUSTED",
        "message": "Document ingestion retry budget exhausted",
        "retryable": False,
    }
}


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_status", ["received", "processing", "parsed", "stored_postgres"])
async def test_exhausted_dispatch_fails_every_pending_status(pending_status: str) -> None:
    """The dead-letter rule: no dead-lettered item remains in a pending state."""
    session = _ScriptedSession([_document_row(status=pending_status, ingestion_state=_IDENTITY)])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.apply_terminal_ingestion_failure(
        document_id=DOCUMENT_ID,
        tenant_id=TENANT,
        document_version=1,
        attempt=0,
        task_id="task-1",
        metadata_=_FAILURE_METADATA,
    )

    assert isinstance(result, Success)
    assert result.unwrap() is True
    update_statement = session.statements[1]
    params = update_statement.compile().params
    assert params["status"] == "failed"
    assert json.loads(params["metadata_"])["ingestion_error"]["code"] == "INGESTION_RETRY_EXHAUSTED"


@pytest.mark.asyncio
async def test_completed_document_is_never_clobbered_by_a_late_dead_letter() -> None:
    for terminal in ("completed", "completed_with_warnings"):
        session = _ScriptedSession([_document_row(status=terminal, ingestion_state=_IDENTITY)])
        repository = DocumentRepository(session)  # type: ignore[arg-type]

        result = await repository.apply_terminal_ingestion_failure(
            document_id=DOCUMENT_ID,
            tenant_id=TENANT,
            document_version=1,
            attempt=0,
            task_id="task-1",
            metadata_=_FAILURE_METADATA,
        )

        assert isinstance(result, Success)
        assert result.unwrap() is False
        assert len(session.statements) == 1, "a completed row is read, never updated"


@pytest.mark.asyncio
async def test_already_failed_document_is_idempotent_under_redelivery() -> None:
    session = _ScriptedSession([_document_row(status="failed", ingestion_state=_IDENTITY)])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.apply_terminal_ingestion_failure(
        document_id=DOCUMENT_ID,
        tenant_id=TENANT,
        document_version=1,
        attempt=0,
        task_id="task-1",
        metadata_=_FAILURE_METADATA,
    )

    assert isinstance(result, Success)
    assert result.unwrap() is False
    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_dead_letter_for_a_re_ingested_identity_is_ignored() -> None:
    """A stale terminal event from attempt 0 must not fail attempt 2's run."""
    current = {**_IDENTITY, "attempt": 2, "task_id": "task-2"}
    session = _ScriptedSession([_document_row(status="processing", ingestion_state=current)])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.apply_terminal_ingestion_failure(
        document_id=DOCUMENT_ID,
        tenant_id=TENANT,
        document_version=1,
        attempt=0,
        task_id="task-1",
        metadata_=_FAILURE_METADATA,
    )

    assert isinstance(result, Success)
    assert result.unwrap() is False
    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_foreign_tenant_document_is_never_touched() -> None:
    session = _ScriptedSession([_document_row(status="received", ingestion_state=_IDENTITY)])
    repository = DocumentRepository(session)  # type: ignore[arg-type]

    result = await repository.apply_terminal_ingestion_failure(
        document_id=DOCUMENT_ID,
        tenant_id="tenant-other",
        document_version=1,
        attempt=0,
        task_id="task-1",
        metadata_=_FAILURE_METADATA,
    )

    assert isinstance(result, Success)
    assert result.unwrap() is False
    assert len(session.statements) == 1
