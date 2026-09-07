"""Celery tasks for unified document ingestion."""

from __future__ import annotations

import asyncio

from app.connections.celery import CeleryTaskPayload, CeleryTaskRegistry, ResilientTask, celery_app
from app.connections.celery_task_names import DOCUMENTS_INGEST
from app.features.documents.service import run_document_ingestion_task
from app.utils import InfrastructureException, logger


class DocumentIngestPayload(CeleryTaskPayload):
    """Typed payload for the unified document ingestion task."""

    document_id: str
    user_id: str
    filename: str
    content_type: str
    object_uri: str


CeleryTaskRegistry.register(DOCUMENTS_INGEST, DocumentIngestPayload)


@celery_app.task(
    name=DOCUMENTS_INGEST,
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
    # The graph never raises for expected ingestion failures; it returns an
    # error state instead. Retryable failures must still raise so Celery
    # retries; permanent ones are returned as failure dicts.
    if result.get("status") == "failed":
        if result.get("error_retryable"):
            self.release_idempotency_processing_lock(idempotency_key)
            raise InfrastructureException(
                detail=str(result.get("error_message") or "Document ingestion failed"),
                error_code=str(result.get("error_code") or "INGESTION_FAILED"),
                retryable=True,
                data={"document_id": document_id},
            )
        self.mark_idempotency_completed(idempotency_key, metadata={"document_id": document_id})
        return result
    self.mark_idempotency_completed(idempotency_key, metadata={"document_id": document_id})
    return result
