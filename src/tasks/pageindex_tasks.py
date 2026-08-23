"""PageIndex ingestion task — declared, deliberately not implemented.

Registration is the point of this module. A task that is declared but absent from
the task application's module list produces the worst diagnostic available: the
dispatch succeeds, the worker reports an unknown name, and the only evidence is
work that never happened. Registered, the same dispatch reaches a body that says
what is wrong. So the name is bound and a payload model is declared, and the body
raises — an explicit not-implemented failure instead of an unknown-task one.
"""

from app.connections.celery import celery_app
from app.connections.celery_registry import CeleryTaskPayload, CeleryTaskRegistry
from app.connections.celery_task_names import PAGEINDEX_INGEST


class PageIndexIngestPayload(CeleryTaskPayload):
    """Typed payload for the PageIndex ingestion task."""

    file_path: str
    user_id: str


CeleryTaskRegistry.register(PAGEINDEX_INGEST, PageIndexIngestPayload)


@celery_app.task(name=PAGEINDEX_INGEST)
def ingest_pageindex_document(file_path: str, user_id: str) -> str:
    """Run PageIndex ingestion in the worker. Not implemented yet."""
    # run in worker - fully async inside via the client above
    message = "PageIndex ingestion is not implemented yet"
    raise NotImplementedError(message)
