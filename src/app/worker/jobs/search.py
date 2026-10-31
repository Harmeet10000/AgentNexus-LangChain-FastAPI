"""Celery execution boundary for site-search ingest jobs.

Mirrors ``app/worker/jobs/documents.py`` structure (typed payload validation
via the catalog, Redis idempotency lock, autoretry on infrastructure
exceptions, note-carrying error handling per EXCEPTION-RULES) with one
deliberate simplification: search ingest needs only Postgres and the embedding
provider, so it provisions a task-local session via ``init_db`` instead of the
document capability's graphiti/ingestion-graph stack.

The task body converts the pipeline's ``Result`` boundary into raises — Celery
entrypoints raise; the autoretry policy owns retry and dead-letter.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.config import get_settings
from app.features.search.application.ingestion import run_search_ingestion_task
from app.observability.logging import logger
from app.shared.jobs.names import SEARCH_INGEST
from app.utils import InfrastructureException, ServiceUnavailableException
from app.worker.celery.app import get_worker_celery_app
from app.worker.celery.task import ResilientTask

celery_app = get_worker_celery_app()

if TYPE_CHECKING:
    from typing import Any


def _redis_task_lock_enabled() -> bool:
    """Return whether the optional Redis-backed outer task lock is configured."""
    return bool(get_settings().REDIS_URL)


@celery_app.task(
    name=SEARCH_INGEST,
    bind=True,
    base=ResilientTask,
    autoretry_for=(
        *ResilientTask.autoretry_for,
        InfrastructureException,
        ServiceUnavailableException,
    ),
    soft_time_limit=1500,
    time_limit=1620,
)
def search_ingest(
    self: ResilientTask,
    *,
    document_id: str,
    tenant_id: str,
    content: str,
    content_hash: str,
    title: str,
    source_uri: str | None = None,
    doc_metadata: dict[str, Any] | None = None,
) -> dict[str, object]:
    del (
        content_hash,
        title,
        source_uri,
        doc_metadata,
    )  # identity fields; chunking needs content only
    idempotency_key = f"search-ingest:{tenant_id}:{document_id}"
    redis_lock_enabled = _redis_task_lock_enabled()
    if redis_lock_enabled and not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"document_id": document_id, "tenant_id": tenant_id},
    ):
        logger.bind(document_id=document_id, task_id=self.request.id).info("search_ingest_locked")
        return {"status": "skipped", "document_id": document_id, "tenant_id": tenant_id}
    try:
        result = asyncio.run(
            run_search_ingestion_task(
                document_id=document_id,
                tenant_id=tenant_id,
                content=content,
            )
        )
    except Exception as exc:
        exc.add_note(f"operation=tasks.search_ingest, document_id={document_id}")
        if redis_lock_enabled:
            self.release_idempotency_processing_lock(idempotency_key)
        raise
    if redis_lock_enabled:
        self.mark_idempotency_completed(idempotency_key, metadata={"document_id": document_id})
    return result
