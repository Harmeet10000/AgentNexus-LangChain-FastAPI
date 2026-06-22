"""Celery tasks for unified document ingestion."""

from __future__ import annotations

import asyncio

from app.connections import celery_app
from app.connections.celery import ResilientTask
from app.connections.celery_registry import CeleryTaskPayload, CeleryTaskRegistry
from app.features.documents.service import run_document_ingestion_task
from app.utils import logger


class DocumentIngestPayload(CeleryTaskPayload):
    """Typed payload for tasks.documents_ingest."""

    document_id: str
    user_id: str
    filename: str
    content_type: str
    object_uri: str


CeleryTaskRegistry.register("tasks.documents_ingest", DocumentIngestPayload)


@celery_app.task(
    name="tasks.documents_ingest",
    bind=True,
    base=ResilientTask,
)
def ingest_document(
    self: ResilientTask,
    *,
    document_id: str,
    user_id: str,
    filename: str,
    content_type: str,
    object_uri: str,
) -> dict[str, object]:
    idempotency_key = f"documents-ingest:{user_id}:{document_id}"
    if not self.acquire_idempotency_lock(idempotency_key, metadata={"document_id": document_id}):
        logger.bind(document_id=document_id, task_id=self.request.id).info(
            "documents_ingest_locked"
        )
        return {"status": "skipped", "document_id": document_id}
    try:
        result = asyncio.run(
            run_document_ingestion_task(
                document_id=document_id,
                user_id=user_id,
                filename=filename,
                content_type=content_type,
                object_uri=object_uri,
            )
        )
    except Exception:
        self.release_idempotency_processing_lock(idempotency_key)
        raise
    self.mark_idempotency_completed(idempotency_key, metadata={"document_id": document_id})
    return result
