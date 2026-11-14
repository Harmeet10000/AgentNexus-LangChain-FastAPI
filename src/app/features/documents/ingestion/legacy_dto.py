"""
Ingestion feature: HTTP upload endpoint that runs IngestionGraph before WS.

Flow:
  POST /ingestion/documents/upload
    → store file (MongoDB GridFS / S3 / local — stub)
    → run IngestionGraph (extract → validate → embed_store)
    → return {doc_id, status, entity_count, clause_count}

  Client then opens WS /agent-saul/ws/{thread_id} with doc_id.
  agent_saul's ingestion_node does a simple SELECT raw_text WHERE doc_id=X.
"""

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel, frozen=True):
    """Deprecated upload contract (F2).

    The legacy graph path still populates the four extraction counts; the canonical
    `POST /documents/upload` command service — which this route now delegates to — queues the
    document and answers before any extraction runs, so it cannot know them. The counts are
    therefore optional with explicit deprecation defaults: `None` means "not computed on this
    path", which is truthful for a queued document and keeps the response model satisfiable
    (before the fix FastAPI raised `ResponseValidationError` AFTER the document was persisted,
    so clients saw a 500 while state had mutated and the deprecation headers never shipped).
    """

    doc_id: str
    status: str
    entity_count: int | None = None
    clause_count: int | None = None
    relationship_count: int | None = None
    dropped_entity_count: int | None = None
    error: str | None = None
