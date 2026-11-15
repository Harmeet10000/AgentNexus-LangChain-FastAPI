"""C7: periodic maintenance/observation tasks (default worker).

Deliberately separate from ``jobs.documents``: the ingestion worker imports
that module wholesale, and a default-queued task registered there would fail
the ingestion profile's completeness check (unexpected task name). The sweep
imports the capability helpers directly — never the ingest task module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from returns.result import Failure

from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
)
from app.observability.degradation import STUCK_DOCUMENTS, record_degradation
from app.observability.logging import logger
from app.shared.jobs.names import DOCUMENTS_STUCK_SWEEP
from app.utils import ServiceUnavailableException
from app.worker.capabilities.documents import (
    get_document_worker_resources,
    run_on_document_worker_loop,
)
from app.worker.celery.app import get_worker_celery_app
from app.worker.celery.task import ResilientTask

if TYPE_CHECKING:
    from app.worker.capabilities.documents import DocumentWorkerResources

celery_app = get_worker_celery_app()


@celery_app.task(name=DOCUMENTS_STUCK_SWEEP, base=ResilientTask)
def sweep_stuck_documents(_self: ResilientTask) -> dict[str, object]:
    """C7: observe every non-terminal document past the stuck-age threshold.

    The per-task safety net (``_warn_if_previous_attempt_stuck``) only fires
    when someone re-attempts ingestion; a document that lost its terminal
    transition and is never retried sat unobserved forever. This beat pass
    makes the observation periodic. It is deliberately OBSERVATION-ONLY —
    log + degradation counter, never a status write — because re-drive
    belongs to the idempotent resume driver, not to a scheduler that cannot
    tell a lost worker from a legitimately long Docling run.
    """
    from app.config import get_settings  # noqa: PLC0415 — task-time settings read

    threshold_seconds = get_settings().DOCUMENTS_STUCK_STATUS_HOURS * 3600
    resources: DocumentWorkerResources = get_document_worker_resources()

    # The documents worker loop owns the only event loop on this process;
    # reuse it rather than spawning a competing one.
    try:
        stuck = run_on_document_worker_loop(
            lambda: _scan(resources=resources, threshold_seconds=threshold_seconds)
        )
    except Exception as exc:  # noqa: BLE001 — a scan outage degrades, never kills the beat pass
        exc.add_note("operation=stuck_document_sweep")
        logger.bind(error=str(exc)).exception("stuck_document_sweep_failed")
        return {"status": "error", "scanned": 0}

    for document_id, stuck_status, age_seconds in stuck:
        logger.bind(
            document_id=document_id,
            stuck_status=stuck_status,
            age_seconds=round(age_seconds, 1),
        ).warning("documents_ingest_stuck_detected")
        record_degradation(STUCK_DOCUMENTS, surface="stuck_sweep", stuck_status=stuck_status)
    if stuck:
        logger.bind(count=len(stuck)).warning("documents_stuck_sweep_report")
    return {"status": "swept", "scanned": len(stuck)}


async def _scan(
    *, resources: DocumentWorkerResources, threshold_seconds: float
) -> list[tuple[str, str, float]]:
    engine, session_local = resources.engine, resources.session_local
    async with engine.connect() as connection, session_local(bind=connection) as session:
        repo = DocumentRepository(session)
        result = await repo.find_stuck_documents(threshold_seconds=threshold_seconds)
        if isinstance(result, Failure):
            raise ServiceUnavailableException(
                detail="Stuck-document scan failed",
                data={"operation": "stuck_document_sweep"},
            )
        await connection.commit()
        return result.unwrap()
