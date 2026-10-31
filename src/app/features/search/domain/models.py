"""Site-search domain vocabulary.

Pure domain types only: no SQLAlchemy, no FastAPI, no I/O. The persistence
shapes for this feature live in ``adapters/persistence/models.py`` because the
task brief places the ORM there (documents keeps its aggregates in ``domain/``;
search follows the brief, not that precedent).
"""

from __future__ import annotations

from enum import StrEnum

#: Redis record written at ingest: outbox event id -> owning tenant/document.
#: It is the authorization source for GET /search/ingest/{task_id} while a
#: task is queued, because the Celery result payload only carries the tenant
#: after a worker has run. Pure string vocabulary — no I/O here.
TASK_OWNER_CACHE_KEY = "search:task_owner:{task_id}"

#: Redis record written by the worker on task entry: outbox event id -> the
#: real Celery task id it was dispatched under. The relay lets Celery assign
#: the id, so polling the event id directly would report PENDING forever;
#: status resolution follows owner record -> this mapping -> document column.
TASK_CELERY_ID_CACHE_KEY = "search:task_celery_id:{task_id}"


class SearchDocumentStatus(StrEnum):
    """Lifecycle of a search document's chunk/embed pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def coerce(cls, raw: object) -> SearchDocumentStatus:
        """Map any stored value onto the enum, defaulting unknowns to PENDING.

        The column is a plain varchar so rolling deploys and legacy rows never
        crash a read; an unrecognized value degrades to the pre-pipeline state.
        """
        try:
            return cls(str(raw))
        except ValueError:
            return cls.PENDING
