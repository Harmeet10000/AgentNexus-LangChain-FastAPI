"""Celery tasks for search ingestion."""

from __future__ import annotations

import asyncio

from app.connections import celery_app
from app.connections.celery import ResilientTask
from app.features.search.service import run_ingestion_task
from app.utils import logger


@celery_app.task(
    name="tasks.search_ingest",
    bind=True,
    base=ResilientTask,
)
def ingest_search_document(
    self: ResilientTask,
    *,
    document_id: str,
    content: str,
    content_hash: str,
    title: str,
    source_uri: str | None = None,
    doc_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Chunk, embed, and persist search data for a queued document."""
    _ = (title, source_uri, doc_metadata)
    idempotency_key = f"search-ingest:{content_hash}"
    if not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"document_id": document_id},
    ):
        logger.bind(document_id=document_id, task_id=self.request.id).info("search_ingest_locked")
        return {
            "status": "skipped",
            "document_id": document_id,
        }

    try:
        result = asyncio.run(
            run_ingestion_task(
                document_id=document_id,
                content=content,
            )
        )
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise

    self.mark_idempotency_completed(
        idempotency_key,
        metadata={"document_id": document_id},
    )
    return result
