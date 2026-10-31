"""Celery execution boundary for site-search ingest jobs.

Mirrors ``app/worker/jobs/documents.py`` structure (typed payload validation
via the catalog, Redis idempotency lock, autoretry on infrastructure
exceptions, note-carrying error handling per EXCEPTION-RULES) with one
deliberate simplification: search ingest needs only Postgres and the embedding
provider, so it provisions a task-local session via ``init_db`` instead of the
document capability's graphiti/ingestion-graph stack.

The task body converts the pipeline's ``Result`` boundary into raises — Celery
entrypoints raise; the autoretry policy owns retry and dead-letter. Retryable
infrastructure failures map to the exceptions in ``autoretry_for``; terminal
failures (configuration drift, uniqueness conflicts) raise
``SearchIngestTerminalError``, which ``dont_autoretry_for`` exempts from the
retry loop — a retry cannot fix them and the document would stay pending
through every backoff.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

from app.config import get_settings
from app.features.search.application.ingestion import (
    SearchIngestTerminalError,
    run_search_ingestion_task,
)
from app.features.search.application.service import bump_tenant_cache_version
from app.features.search.domain.models import TASK_CELERY_ID_CACHE_KEY
from app.observability.logging import logger
from app.shared.jobs.names import SEARCH_INGEST
from app.utils import InfrastructureException, ServiceUnavailableException
from app.worker.celery.app import get_worker_celery_app
from app.worker.celery.task import ResilientTask
from app.worker.execution.runtime import run

celery_app = get_worker_celery_app()

if TYPE_CHECKING:
    from typing import Any

    from redis.asyncio import Redis as AsyncRedis


def _redis_task_lock_enabled() -> bool:
    """Return whether the optional Redis-backed outer task lock is configured."""
    return bool(get_settings().REDIS_URL)


async def record_celery_task_id(
    redis_client: AsyncRedis[str],
    *,
    event_id: str | None,
    task_id: str | None,
) -> None:
    """Bind the real Celery dispatch id to the outbox event id (best effort).

    The relay lets Celery assign the task id, so the id the API returned (the
    outbox event id) would poll as PENDING forever. The worker writes the
    mapping once it owns the processing, so status resolution can poll the
    real id. A write failure degrades to a log line — the durable document
    status column, not this cache, is the terminal truth.
    """
    if not event_id or not task_id:
        return
    try:
        # The worker client is redis.asyncio: the set MUST be awaited, or the
        # coroutine is discarded and the mapping is never stored (which left
        # status polling stuck on QUEUED in production review).
        await redis_client.set(
            TASK_CELERY_ID_CACHE_KEY.format(task_id=event_id),
            str(task_id),
            ex=get_settings().SEARCH_TASK_OWNER_TTL_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — the mapping is an accelerator, not truth
        exc.add_note(f"operation=search_task_celery_id_record, event_id={event_id}")
        exc.add_note(traceback.format_exc())
        logger.bind(event_id=event_id, task_id=task_id).warning(
            "search_task_celery_id_record_failed"
        )


@celery_app.task(
    name=SEARCH_INGEST,
    bind=True,
    base=ResilientTask,
    autoretry_for=(
        *ResilientTask.autoretry_for,
        InfrastructureException,
        ServiceUnavailableException,
    ),
    dont_autoretry_for=(SearchIngestTerminalError,),
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
    event_id: str | None = None,
) -> dict[str, object]:
    del (
        content_hash,
        title,
        source_uri,
        doc_metadata,
    )  # identity fields; chunking needs content only
    # Redis availability gates EVERYTHING cache-shaped: with an empty REDIS_URL
    # there is no client to construct, so the lock check must run before any
    # get_redis_client() call.
    redis_lock_enabled = _redis_task_lock_enabled()
    idempotency_key = f"search-ingest:{tenant_id}:{document_id}"
    if redis_lock_enabled and not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={"document_id": document_id, "tenant_id": tenant_id},
    ):
        logger.bind(document_id=document_id, task_id=self.request.id).info("search_ingest_locked")
        return {"status": "skipped", "document_id": document_id, "tenant_id": tenant_id}

    async def _run_owned_pipeline() -> dict[str, object]:
        # Only the worker that won the idempotency lock records its dispatch
        # id: a skipped redelivery must not overwrite the winner's mapping.
        if redis_lock_enabled:
            await record_celery_task_id(
                self.get_redis_client(),
                event_id=event_id,
                task_id=self.request.id,
            )
        result = await run_search_ingestion_task(
            document_id=document_id,
            tenant_id=tenant_id,
            content=content,
            event_id=event_id,
        )
        # Fresh ingest must not serve stale cached searches for the remainder
        # of the cache TTL: bump the tenant's cache version so every digest
        # computed afterwards misses (the old keys simply expire).
        if redis_lock_enabled:
            await bump_tenant_cache_version(self.get_redis_client(), tenant_id=tenant_id)
        return result

    try:
        # The worker's cached async Redis client is bound to the persistent
        # per-process loop; running the pipeline on a fresh asyncio.run loop
        # would reuse open connections across loops.  runtime.run retains one
        # loop between tasks, so every Redis call — lock, task-id link, cache
        # bump — stays on the same event loop.
        result = run(_run_owned_pipeline())
    except Exception as exc:
        exc.add_note(f"operation=tasks.search_ingest, document_id={document_id}")
        if redis_lock_enabled:
            self.release_idempotency_processing_lock(idempotency_key)
        raise
    if redis_lock_enabled:
        self.mark_idempotency_completed(idempotency_key, metadata={"document_id": document_id})
    return result
