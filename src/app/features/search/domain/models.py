"""Site-search domain vocabulary.

Pure domain types only: no SQLAlchemy, no FastAPI, no I/O. The persistence
shapes for this feature live in ``adapters/persistence/models.py`` because the
task brief places the ORM there (documents keeps its aggregates in ``domain/``;
search follows the brief, not that precedent).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict

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

#: Per-tenant counter folded into every response-cache digest. The worker
#: bumps it on successful ingestion, so a fresh ingest never serves the stale
#: cached results of the corpus it just changed — without a pattern SCAN.
CACHE_VERSION_KEY = "search:cache_version:{tenant_id}"


class SearchDocument(BaseModel):
    """Persistence-neutral document state used by search use cases.

    The PostgreSQL row is an adapter detail.  Returning this immutable value
    keeps SQLAlchemy instrumentation and session state out of the application
    interface while preserving every field the use cases actually read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    tenant_id: str
    source_uri: str | None = None
    title: str
    content_hash: str
    status: str
    doc_metadata: dict[str, object]
    ingested_at: datetime
    updated_at: datetime


class SearchChunkUpsert(BaseModel):
    """One provider-embedded chunk ready for persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    document_id: UUID
    tenant_id: str
    chunk_index: int
    content: str
    embedding: list[float]
    chunk_metadata: dict[str, object]


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
