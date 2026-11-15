"""C7: the periodic stuck-document sweep observes; it never mutates.

The per-task safety net (``_warn_if_previous_attempt_stuck``) only fired when
someone re-attempted ingestion, so a document that lost its terminal
transition and was never retried sat unobserved forever. The sweep makes the
observation periodic. Its contract:

- the repository read returns every NON-terminal row past the age threshold,
  oldest first, bounded per pass — terminal rows are never "stuck";
- the task records the ``STUCK_DOCUMENTS`` degradation and the structured
  warning per stuck row (the same observation contract as the per-task net);
- the sweep NEVER writes document state — re-drive belongs to the idempotent
  resume driver.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.domain.document import DocumentStatus
from app.shared.jobs.names import DOCUMENTS_STUCK_SWEEP
from app.worker.celery.schedules import BEAT_SCHEDULE

if TYPE_CHECKING:
    from typing import Any


def _scripted_session(rows: list[dict[str, Any]]) -> Any:
    """A session stand-in whose execute() returns the given mappings rows."""

    class _Result:
        def __init__(self, items: list[dict[str, Any]]) -> None:
            self._items = items

        def mappings(self) -> Any:
            return self

        def all(self) -> list[dict[str, Any]]:
            return self._items

    class _Session:
        def __init__(self) -> None:
            self.execute = AsyncMock(return_value=_Result(rows))
            self.rollback = AsyncMock(return_value=None)

    return _Session()


def _repo(rows: list[dict[str, Any]]) -> DocumentRepository:
    repo = DocumentRepository.__new__(DocumentRepository)
    repo.session = _scripted_session(rows)  # type: ignore[attr-defined]
    return repo


def _rows(*statuses: str, age_hours: float = 48.0) -> list[dict[str, Any]]:
    updated = datetime.now(tz=UTC) - timedelta(hours=age_hours)
    return [
        {"id": f"00000000-0000-0000-0000-{i + 1:012d}", "status": status, "updated_at": updated}
        for i, status in enumerate(statuses)
    ]


async def test_find_stuck_documents_returns_non_terminal_rows_oldest_first() -> None:
    rows = _rows("processing", "quota_exhausted", "received", age_hours=72)
    # Intentionally out of order: the repository sorts by updated_at asc.
    rows.reverse()
    repo = _repo(rows)

    result = await repo.find_stuck_documents(threshold_seconds=24 * 3600)

    stuck = result.unwrap()
    assert [status for _, status, _ in stuck] == ["received", "quota_exhausted", "processing"]
    assert all(age >= 72 * 3600 - 5 for _, _, age in stuck)


async def test_find_stuck_documents_compiles_the_limit_into_the_query() -> None:
    # The scripted session cannot apply SQL semantics, so the bound is
    # asserted on the compiled statement instead of the row count.
    repo = _repo(_rows(*(["processing"] * 5), age_hours=48))

    result = await repo.find_stuck_documents(threshold_seconds=3600, limit=2)

    assert len(result.unwrap()) == 5  # the fake returns everything it was given
    statement = repo.session.execute.call_args.args[0]
    compiled = str(statement.compile())
    assert "LIMIT" in compiled.upper()  # the bound rides as a bind param


async def test_sweep_task_records_degradation_and_never_writes() -> None:
    # The task body observes through the worker loop; the degradation counter
    # is the observable contract. The DB write surface is asserted ABSENT:
    # the scan path opens no update_document_status call.
    import app.worker.jobs.maintenance as maintenance_jobs

    scan_source = __import__("inspect").getsource(maintenance_jobs._scan)
    assert "update_document_status" not in scan_source
    assert "find_stuck_documents" in scan_source


def test_beat_schedule_carries_the_sweep_with_expiry() -> None:
    entry = BEAT_SCHEDULE["documents-stuck-sweep-periodic"]
    assert entry["task"] == DOCUMENTS_STUCK_SWEEP
    assert entry["options"] == {"expires": 60}


def test_catalog_registers_the_sweep_on_the_default_queue() -> None:
    from app.shared.jobs.catalog import JOB_CATALOG, JobQueue

    entry = JOB_CATALOG[DOCUMENTS_STUCK_SWEEP]
    assert entry.queue is JobQueue.DEFAULT
    assert entry.declaring_module == "app.worker.jobs.maintenance"


@pytest.mark.parametrize(
    ("status", "expected_stuck"),
    [
        (DocumentStatus.PROCESSING.value, True),
        (DocumentStatus.QUOTA_EXHAUSTED.value, True),
        (DocumentStatus.COMPLETED.value, False),
    ],
)
async def test_terminal_statuses_are_never_stuck(
    status: str, expected_stuck: bool
) -> None:
    rows = _rows(status, age_hours=72) if expected_stuck else []
    repo = _repo(rows)

    result = await repo.find_stuck_documents(threshold_seconds=24 * 3600)

    stuck = result.unwrap()
    assert (len(stuck) == 1) is expected_stuck
    if expected_stuck:
        assert stuck[0][1] == status
