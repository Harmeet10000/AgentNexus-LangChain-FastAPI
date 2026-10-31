"""Worker-side ingest pipeline: width guard, batched upserts, terminal states.

The pipeline is the Celery raising boundary (EXCEPTION-RULES): a configuration
drift between the embedding producer width and the stored column must raise a
typed configuration error instead of being retried, chunks must upsert per
embedding batch (never one giant INSERT), and even a zero-chunk document must
record a terminal status the tenant can poll.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from returns.result import Success

from app.features.search.application import ingestion as ingestion_module
from app.features.search.application.ingestion import (
    SearchIngestTerminalError,
    process_search_ingestion_document,
)
from app.features.search.domain.errors import SearchConflictError
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from typing import Any


TENANT = "tenant-1"
DOC_ID = "00000000-0000-0000-0000-0000000000aa"


class _PipelineRepository:
    """Records upsert batches and status writes without a database."""

    def __init__(self):
        self.upsert_batches: list[list[dict[str, Any]]] = []
        self.statuses: list[tuple[str, str]] = []
        self.analyze_calls = 0

    async def upsert_chunks(self, rows: list[dict[str, Any]]) -> Any:
        self.upsert_batches.append(rows)
        return Success(None)

    async def analyze_chunks(self) -> Any:
        self.analyze_calls += 1
        return Success(None)

    async def set_document_status(self, *, document_id: str, status: str) -> Any:
        self.statuses.append((document_id, status))
        return Success(None)


class _FakeSettings:
    """Small geometry so a handful of tokens spans multiple batches."""

    SEARCH_INGEST_CHUNK_SIZE = 2
    SEARCH_INGEST_CHUNK_OVERLAP = 0
    SEARCH_INGEST_EMBEDDING_BATCH_SIZE = 2
    SEARCH_ANALYZE_THRESHOLD_CHUNKS = 10_000


def _install_pipeline(monkeypatch: pytest.MonkeyPatch) -> _PipelineRepository:
    """Wire the small-geometry settings and a stubbed embedder; return the repo."""
    repo = _PipelineRepository()
    monkeypatch.setattr(ingestion_module, "get_settings", _FakeSettings)

    async def _fake_embed(texts: list[str], *, task_type: Any) -> list[list[float]]:
        return [[0.0, 0.1] for _ in texts]

    monkeypatch.setattr(ingestion_module, "embed_texts", _fake_embed)
    return repo


@pytest.mark.asyncio
async def test_chunks_are_upserted_per_embedding_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five tokens at chunk-size 2 → three chunks → batches of 2 and 1."""
    pipeline = _install_pipeline(monkeypatch)
    result = await process_search_ingestion_document(
        repo=pipeline,  # type: ignore[arg-type]
        document_id=DOC_ID,
        tenant_id=TENANT,
        content="a b c d e",
    )

    assert result["status"] == "completed"
    assert result["chunk_count"] == 3
    assert [len(batch) for batch in pipeline.upsert_batches] == [2, 1]
    # Every row is tenant-stamped; all rows bind the same document.
    for batch in pipeline.upsert_batches:
        for row in batch:
            assert row["tenant_id"] == TENANT
            assert str(row["document_id"]) == DOC_ID
    assert pipeline.statuses == [(DOC_ID, "completed")]


@pytest.mark.asyncio
async def test_empty_content_records_a_completed_zero_chunk_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only content must not strand the document as pending."""
    pipeline = _install_pipeline(monkeypatch)
    result = await process_search_ingestion_document(
        repo=pipeline,  # type: ignore[arg-type]
        document_id=DOC_ID,
        tenant_id=TENANT,
        content="   \n\t ",
    )

    assert result == {
        "status": "completed",
        "document_id": DOC_ID,
        "event_id": None,
        "chunk_count": 0,
    }
    assert pipeline.upsert_batches == []
    assert pipeline.statuses == [(DOC_ID, "completed")]


@pytest.mark.asyncio
async def test_width_drift_raises_the_terminal_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A column/setting disagreement is terminal — the task must not retry it."""

    def _mismatch(stored_dim: int) -> tuple[int, int] | None:
        return (stored_dim, 1536)

    monkeypatch.setattr(ingestion_module, "stored_width_mismatch", _mismatch)
    pipeline = _install_pipeline(monkeypatch)

    with pytest.raises(SearchIngestTerminalError, match=r"search_chunks\.embedding"):
        await process_search_ingestion_document(
            repo=pipeline,  # type: ignore[arg-type]
            document_id=DOC_ID,
            tenant_id=TENANT,
            content="a b c",
        )
    # Nothing was written before the guard fired.
    assert pipeline.upsert_batches == []
    assert pipeline.statuses == []


def test_to_raising_exception_maps_by_retryability() -> None:
    """Infrastructure failures retry; everything else is terminal (CR MINOR)."""
    from app.features.search.domain.errors import SearchInfrastructureError

    error = SearchInfrastructureError(message="database unavailable", source="test")
    translated = ingestion_module._to_raising_exception(error)

    assert isinstance(translated, ServiceUnavailableException)
    non_infra = SearchConflictError(message="duplicate", source="test")
    terminal = ingestion_module._to_raising_exception(non_infra)
    assert isinstance(terminal, SearchIngestTerminalError)
    assert not isinstance(terminal, ServiceUnavailableException)
