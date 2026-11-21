"""Import-safe composition descriptor for the documents slice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.shared.module import Module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any


def build_ingestion_consumers() -> Sequence[Any]:
    """Bind ingestion-queue consumers only inside an ingestion worker."""
    from .adapters.inbound.worker.ingestion_tasks import (  # noqa: PLC0415
        ingest_document,
        ingest_document_terminal,
    )

    return (ingest_document, ingest_document_terminal)


def build_maintenance_consumers() -> Sequence[Any]:
    """Bind the default-queue document observer only inside a worker."""
    from .adapters.inbound.worker.maintenance_tasks import (  # noqa: PLC0415
        sweep_stuck_documents,
    )

    return (sweep_stuck_documents,)


def build_schedules() -> Mapping[str, Any]:
    """Build the observation-only stuck-document sweep schedule."""
    from celery.schedules import crontab  # noqa: PLC0415

    from app.shared.jobs.names import DOCUMENTS_STUCK_SWEEP  # noqa: PLC0415

    return {
        "documents-stuck-sweep-periodic": {
            "task": DOCUMENTS_STUCK_SWEEP,
            # The sweep only observes stale non-terminal work. Re-drive remains
            # with the idempotent resume driver, never with the scheduler.
            "schedule": crontab(minute="*/15"),
            "options": {"expires": 60},
        }
    }


MODULE = Module(
    name="documents",
    consumers=(build_ingestion_consumers, build_maintenance_consumers),
    schedules=(build_schedules,),
)
