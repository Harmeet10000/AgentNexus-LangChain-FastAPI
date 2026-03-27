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

from __future__ import annotations

# === dto.py content ===
from pydantic import BaseModel


class DocumentUploadResponse(BaseModel, frozen=True):
    doc_id: str
    status: str
    entity_count: int
    clause_count: int
    relationship_count: int
    dropped_entity_count: int
    error: str | None = None
