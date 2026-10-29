"""Celery tasks for unified document ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from returns.result import Failure

from app.config import get_settings
from app.events import with_terminal_outbox
from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.features.documents.application.commands.upload_document import run_document_ingestion_task
from app.observability.logging import logger
from app.shared.jobs.names import DOCUMENTS_INGEST, DOCUMENTS_INGEST_TERMINAL
from app.utils import InfrastructureException, ServiceUnavailableException
from app.worker.capabilities.documents import (
    get_document_worker_resources,
    run_on_document_worker_loop,
)
from app.worker.celery.app import get_worker_celery_app
from app.worker.celery.task import ResilientTask

celery_app = get_worker_celery_app()

if TYPE_CHECKING:
    from typing import Any

    from app.worker.capabilities.documents import (
        DocumentWorkerResources,
    )


class DocumentIngestionTask(ResilientTask):
    """Celery lifecycle bridge for an exhausted ingestion retry sequence.

    The task body deliberately leaves retryable failures non-terminal.  Celery
    invokes ``on_failure`` only after the retry policy has been exhausted; the
    bridge then uses the worker's already-provisioned loop to apply the exact
    attempt identity and stage the terminal event in the same transaction.
    """

    abstract = True

    @override
    def on_failure(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        super().on_failure(exc, task_id, args, kwargs, einfo)
        request_retries = int(getattr(self.request, "retries", 0) or 0)
        # ``autoretry_for`` passes the class ``retry_kwargs`` to
        # ``Task.retry``.  That value (5 in production) is authoritative over
        # Celery's Task.max_retries default (3); using the latter would bridge
        # an intermediate retry as terminal.
        configured_max = self.retry_kwargs.get("max_retries")
        if configured_max is None:
            configured_max = getattr(self, "max_retries", 0)
        # ``on_failure`` also receives permanent exceptions.  Only exceptions
        # covered by Celery's autoretry policy should defer the terminal
        # bridge while budget remains; an unexpected task error must still
        # leave the document in a terminal state immediately.
        autoretry_types = tuple(
            {
                *ResilientTask.autoretry_for,
                *getattr(self, "autoretry_for", ()),
            }
        )
        if isinstance(exc, autoretry_types) and request_retries < int(configured_max):
            return
        payload = dict(kwargs or {})
        document_id = payload.get("document_id")
        user_id = payload.get("user_id")
        if not isinstance(document_id, str) or not isinstance(user_id, str):
            return
        tenant_id = payload.get("tenant_id")
        resolved_tenant_id = (
            tenant_id if isinstance(tenant_id, str) and tenant_id else f"personal:{user_id}"
        )
        document_version = payload.get("document_version", 1)
        if not isinstance(document_version, int) or isinstance(document_version, bool):
            document_version = 1
        error_code = str(getattr(exc, "error_code", "INGESTION_RETRY_EXHAUSTED"))
        error_message = "Document ingestion retry budget exhausted"
        try:
            resources = get_document_worker_resources()
            run_on_document_worker_loop(
                lambda: _persist_exhausted_terminal_failure(
                    resources=resources,
                    document_id=document_id,
                    user_id=user_id,
                    tenant_id=resolved_tenant_id,
                    document_version=document_version,
                    attempt=request_retries,
                    task_id=task_id,
                    error_code=error_code,
                    error_message=error_message,
                )
            )
        except Exception as bridge_exc:  # noqa: BLE001 — failure hook must not mask Celery outcome
            bridge_exc.add_note(
                f"document_id={document_id}, operation=terminal_ingestion_failure_bridge"
            )
            logger.bind(document_id=document_id, task_id=task_id).exception(
                "Exhausted ingestion failure bridge failed"
            )


def _redis_task_lock_enabled() -> bool:
    """Return whether the optional Redis-backed outer task lock is configured."""
    return bool(get_settings().REDIS_URL)


@celery_app.task(
    name=DOCUMENTS_INGEST,
    bind=True,
    base=DocumentIngestionTask,
    autoretry_for=(
        *ResilientTask.autoretry_for,
        InfrastructureException,
        ServiceUnavailableException,
    ),
    soft_time_limit=1500,
    time_limit=1620,
)
def ingest_document(
    self: ResilientTask,
    *,
    document_id: str,
    user_id: str,
    tenant_id: str | None = None,
    document_version: int = 1,
    filename: str,
    content_type: str,
    object_uri: str,
) -> dict[str, object]:
    resolved_tenant_id = tenant_id or f"personal:{user_id}"
    # Version is part of the work identity.  A re-ingest of the same document
    # must not be suppressed by the Redis lock/completion marker for v1.
    idempotency_key = f"documents-ingest:{resolved_tenant_id}:{document_id}:v{document_version}"
    redis_lock_enabled = _redis_task_lock_enabled()
    if redis_lock_enabled and not self.acquire_idempotency_lock(
        idempotency_key,
        metadata={
            "document_id": document_id,
            "tenant_id": resolved_tenant_id,
            "document_version": document_version,
        },
    ):
        logger.bind(document_id=document_id, task_id=self.request.id).info(
            "documents_ingest_locked"
        )
        return {"status": "skipped", "document_id": document_id}
    try:
        resources: DocumentWorkerResources = get_document_worker_resources()
        result = run_on_document_worker_loop(
            lambda: run_document_ingestion_task(
                document_id=document_id,
                user_id=user_id,
                tenant_id=resolved_tenant_id,
                document_version=document_version,
                filename=filename,
                content_type=content_type,
                object_uri=object_uri,
                task_id=self.request.id,
                attempt=self.request.retries,
                graph=resources.ingestion_graph,
                engine=resources.engine,
                session_local=resources.session_local,
                idempotency=resources.idempotency,
            )
        )
    except Exception as exc:
        exc.add_note(f"operation=documents.ingest, document_id={document_id}")
        if redis_lock_enabled:
            self.release_idempotency_processing_lock(idempotency_key)
        raise
    # The graph never raises for expected ingestion failures; it returns an
    # error state instead. Retryable failures must still raise so Celery
    # retries; permanent ones are returned as failure dicts.
    if result.get("status") == "skipped":
        if redis_lock_enabled:
            self.release_idempotency_processing_lock(idempotency_key)
        return result
    if result.get("status") == "failed":
        if result.get("error_retryable"):
            if redis_lock_enabled:
                self.release_idempotency_processing_lock(idempotency_key)
            raise InfrastructureException(
                detail=str(result.get("error_message") or "Document ingestion failed"),
                error_code=str(result.get("error_code") or "INGESTION_FAILED"),
                retryable=True,
                data={"document_id": document_id},
            )
        if redis_lock_enabled:
            self.mark_idempotency_completed(idempotency_key, metadata={"document_id": document_id})
        return result
    if redis_lock_enabled:
        self.mark_idempotency_completed(idempotency_key, metadata={"document_id": document_id})
    return result


@celery_app.task(
    name=DOCUMENTS_INGEST_TERMINAL,
    bind=True,
    base=ResilientTask,
    autoretry_for=(
        *ResilientTask.autoretry_for,
        InfrastructureException,
        ServiceUnavailableException,
    ),
)
def ingest_document_terminal(
    self: ResilientTask,
    *,
    document_id: str,
    user_id: str,
    tenant_id: str,
    status: str,
    document_version: int = 1,
    attempt: int = 0,
    task_id: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, object]:
    """Consume a terminal event and persist its failed document status.

    The outbox relay can dead-letter an ingestion before its worker ever
    starts.  This consumer is therefore the durable failed-status fallback;
    duplicate deliveries only repeat the same idempotent status update.
    """
    effective_task_id = task_id or self.request.id
    if status == "failed":
        resources: DocumentWorkerResources = get_document_worker_resources()
        applied = run_on_document_worker_loop(
            lambda: _persist_terminal_failure(
                resources=resources,
                document_id=document_id,
                tenant_id=tenant_id,
                error_code=error_code,
                error_message=error_message,
                document_version=document_version,
                attempt=attempt,
                task_id=effective_task_id,
            )
        )
    logger.bind(
        document_id=document_id,
        user_id=user_id,
        tenant_id=tenant_id,
        status=status,
        error_code=error_code,
        applied=applied if status == "failed" else None,
    ).info("documents_ingest_terminal")
    return {"status": status, "document_id": document_id}


async def _persist_terminal_failure(
    *,
    resources: DocumentWorkerResources,
    document_id: str,
    tenant_id: str,
    error_code: str | None,
    error_message: str | None,
    document_version: int,
    attempt: int,
    task_id: str | None,
) -> bool:
    """Persist the relay's failed terminal outcome on the worker loop."""
    async with resources.session_local() as session:
        repository = DocumentRepository(session)
        result = await repository.apply_terminal_ingestion_failure(
            document_id=document_id,
            tenant_id=tenant_id,
            document_version=document_version,
            attempt=attempt,
            task_id=task_id or "",
            metadata_={
                "ingestion_error": {
                    "code": error_code or "INGESTION_TERMINAL_FAILED",
                    "message": error_message or "Document ingestion failed",
                    "retryable": False,
                },
                "ingestion_state": {
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "document_version": document_version,
                    "attempt": attempt,
                    "task_id": task_id or "",
                },
            },
        )
        if isinstance(result, Failure):
            error = result.failure()
            raise ServiceUnavailableException(detail=error.message)
        await session.commit()
        return result.unwrap()


async def _persist_exhausted_terminal_failure(
    *,
    resources: DocumentWorkerResources,
    document_id: str,
    user_id: str,
    tenant_id: str,
    document_version: int,
    attempt: int,
    task_id: str,
    error_code: str,
    error_message: str,
) -> bool:
    """Atomically persist one exhausted attempt and its terminal outbox event."""
    async with resources.session_local() as session:
        repository = DocumentRepository(session)
        result = await repository.apply_terminal_ingestion_failure(
            document_id=document_id,
            tenant_id=tenant_id,
            document_version=document_version,
            attempt=attempt,
            task_id=task_id,
            metadata_={
                "ingestion_error": {
                    "code": error_code,
                    "message": error_message,
                    "retryable": False,
                },
                "ingestion_state": {
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "document_version": document_version,
                    "attempt": attempt,
                    "task_id": task_id,
                },
            },
        )
        if isinstance(result, Failure):
            error = result.failure()
            raise ServiceUnavailableException(detail=error.message)
        applied = result.unwrap()
        if applied:
            await with_terminal_outbox(
                session,
                document_id=document_id,
                payload={
                    "document_id": document_id,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "status": "failed",
                    "document_version": document_version,
                    "attempt": attempt,
                    "task_id": task_id,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
        await session.commit()
        return applied
