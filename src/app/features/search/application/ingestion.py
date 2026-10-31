"""Site-search chunk/embed ingestion pipeline (worker-side).

Ported from the legacy ``logs/search/service.py`` process functions with the
modern seams: typed job payload fields instead of raw kwargs, the
settings-driven shared Gemini embedding client (``app.ai.embeddings``) instead
of a per-call module-level client, and tenant-scoped chunk rows.

Celery entrypoints raise (EXCEPTION-RULES); retryable infrastructure failures
propagate to the task's autoretry policy, while terminal failures raise
:data:`SearchIngestTerminalError` — the task's ``dont_autoretry_for`` exempts
it, because a retry cannot fix a configuration drift or a constraint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from returns.result import Failure

from app.ai.embeddings import EmbeddingTaskType, embed_texts
from app.config import get_settings
from app.features.search.adapters.persistence.models import SEARCH_EMBEDDING_DIM
from app.features.search.adapters.persistence.postgres.search_repository import (
    SearchRepository,
    build_chunk_rows,
)
from app.features.search.domain.chunking import chunk_text
from app.features.search.domain.errors import SearchInfrastructureError
from app.features.search.domain.models import SearchDocumentStatus
from app.observability.logging import logger
from app.utils import ServiceUnavailableException
from app.utils.embedding import stored_width_mismatch, width_mismatch_detail

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.features.search.domain.chunking import TextChunk
    from app.features.search.domain.errors import SearchError


class SearchIngestTerminalError(Exception):
    """A terminal search-ingest failure: retrying cannot change the outcome.

    Configuration drift (the stored vector width disagreeing with the
    configured producer) and non-retryable constraint failures land here. The
    ``tasks.search_ingest`` task's ``dont_autoretry_for`` exempts this class,
    so it dead-letters immediately instead of spinning through backoff while
    the document stays pending.
    """


def _to_raising_exception(error: SearchError) -> Exception:
    """Translate the typed search error union at the Celery raising boundary.

    ``FeatureError`` is a Pydantic model, never raised (EXCEPTION-RULES): the
    task body must raise real exceptions so the autoretry policy can classify
    them. Infrastructure-class failures map to the retryable 503 exception the
    task's ``autoretry_for`` names; everything else (validation, conflict,
    not-found — all ``retryable=False`` by construction) is terminal, because
    the same payload redelivered to the same table can only collide again.
    """
    if isinstance(error, SearchInfrastructureError):
        return ServiceUnavailableException(detail=error.message, data=error.details)
    return SearchIngestTerminalError(error.message)


async def process_search_ingestion_document(
    *,
    repo: SearchRepository,
    document_id: str,
    tenant_id: str,
    content: str,
    event_id: str | None = None,
    doc_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Chunk, batch-embed, and upsert search chunks for one document.

    Idempotent end to end: chunk indexes are stable for identical content, and
    the (document_id, chunk_index) upsert makes re-runs converge to the same
    rows rather than duplicating them. Embedding and upsert happen per batch,
    in one transaction, so a huge document never materializes a single giant
    INSERT and memory stays bounded by SEARCH_INGEST_EMBEDDING_BATCH_SIZE.
    """
    settings = get_settings()
    # Fail fast when the configured producer width disagrees with the width
    # this relation stores: a retry loop cannot fix a column/setting drift,
    # so this raises the terminal (non-retried) exception class directly.
    mismatch = stored_width_mismatch(SEARCH_EMBEDDING_DIM)
    if mismatch is not None:
        stored, expected = mismatch
        raise SearchIngestTerminalError(
            width_mismatch_detail(stored, expected, relation="search_chunks.embedding")
        )
    chunks = chunk_text(
        content,
        chunk_size=settings.SEARCH_INGEST_CHUNK_SIZE,
        chunk_overlap=settings.SEARCH_INGEST_CHUNK_OVERLAP,
    )

    if not chunks:
        # Even a zero-chunk ingest is a terminal state the tenant can poll:
        # record it instead of leaving the row pending forever.
        status_result = await repo.set_document_status(
            document_id=document_id, status=SearchDocumentStatus.COMPLETED.value
        )
        if isinstance(status_result, Failure):
            raise _to_raising_exception(status_result.failure())
        return {
            "status": "completed",
            "document_id": document_id,
            "event_id": event_id,
            "chunk_count": 0,
        }

    chunk_count = 0
    for batch in _batched(chunks, settings.SEARCH_INGEST_EMBEDDING_BATCH_SIZE):
        embeddings = await embed_texts(
            [chunk.content for chunk in batch],
            task_type=EmbeddingTaskType.DOCUMENT,
        )
        batch_rows = [
            {
                "id": str(uuid4()),
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": embedding,
                "chunk_metadata": {
                    **(doc_metadata or {}),
                    "token_count": chunk.token_count,
                },
            }
            for chunk, embedding in zip(batch, embeddings, strict=True)
        ]
        # One INSERT per embedding batch, all inside the caller's transaction:
        # the whole document either lands or rolls back, but no statement ever
        # carries every chunk of a huge document at once.
        upsert_result = await repo.upsert_chunks(
            build_chunk_rows(
                document_id=document_id,
                tenant_id=tenant_id,
                chunks=batch_rows,
            )
        )
        if isinstance(upsert_result, Failure):
            raise _to_raising_exception(upsert_result.failure())
        chunk_count += len(batch_rows)

    if chunk_count > settings.SEARCH_ANALYZE_THRESHOLD_CHUNKS:
        analyze_result = await repo.analyze_chunks()
        if isinstance(analyze_result, Failure):
            raise _to_raising_exception(analyze_result.failure())

    status_result = await repo.set_document_status(
        document_id=document_id, status=SearchDocumentStatus.COMPLETED.value
    )
    if isinstance(status_result, Failure):
        raise _to_raising_exception(status_result.failure())

    logger.bind(
        document_id=document_id,
        tenant_id=tenant_id,
        event_id=event_id,
        chunk_count=chunk_count,
    ).info("search_ingest_processed")
    return {
        "status": "completed",
        "document_id": document_id,
        "tenant_id": tenant_id,
        "event_id": event_id,
        "chunk_count": chunk_count,
        "embedding_dim": SEARCH_EMBEDDING_DIM,
    }


async def run_search_ingestion_task(
    *,
    document_id: str,
    tenant_id: str,
    content: str,
    event_id: str | None = None,
    doc_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create a task-local async DB session and execute search ingestion."""
    from app.connections import init_db  # noqa: PLC0415 — keeps the web import graph worker-free

    engine, session_local = await init_db()
    try:
        async with session_local() as session, session.begin():
            repo = SearchRepository(session)
            return await process_search_ingestion_document(
                repo=repo,
                document_id=document_id,
                tenant_id=tenant_id,
                content=content,
                event_id=event_id,
                doc_metadata=doc_metadata,
            )
    finally:
        await engine.dispose()


def _batched(values: Sequence[TextChunk], batch_size: int) -> list[Sequence[TextChunk]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]
