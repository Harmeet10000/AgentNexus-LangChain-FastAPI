"""
IngestionService: runs IngestionGraph for a given uploaded document.

Called by the HTTP router. Non-streaming — awaits completion before returning.
If IngestionGraph fails, returns an error status (not a 500) so the client
can retry or show a user-facing error.
"""

from __future__ import annotations

from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from app.utils import logger

from .dto import DocumentUploadResponse


class IngestionService:
    def __init__(self, ingestion_graph: CompiledStateGraph) -> None:
        self._graph = ingestion_graph

    async def ingest_document(
        self,
        raw_bytes: bytes,
        user_id: str,
        filename: str,
        source: str,
        document_type: str = "unknown",
        jurisdiction: str = "India",
        doc_id: str | None = None,
    ) -> DocumentUploadResponse:
        resolved_doc_id = doc_id or str(uuid4())
        thread_id = str(uuid4())  # ingestion gets its own thread_id

        log = logger.bind(
            doc_id=resolved_doc_id,
            user_id=user_id,
            document_type=document_type,
        )
        log.info("ingestion_started")

        initial_state = {
            "doc_id": resolved_doc_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "raw_bytes": raw_bytes,
            "filename": filename,
            "source": source,
            "document_type": document_type,
            "jurisdiction": jurisdiction,
        }

        try:
            result = await self._graph.ainvoke(initial_state)
        except Exception as exc:
            log.exception("ingestion_graph_failed", error=str(exc))
            return DocumentUploadResponse(
                doc_id=resolved_doc_id,
                status="failed",
                entity_count=0,
                clause_count=0,
                relationship_count=0,
                dropped_entity_count=0,
                error=str(exc),
            )

        if result.get("error"):
            log.error("ingestion_error", error=result["error"])
            return DocumentUploadResponse(
                doc_id=resolved_doc_id,
                status="failed",
                entity_count=0,
                clause_count=0,
                relationship_count=0,
                dropped_entity_count=result.get("dropped_entity_count", 0),
                error=result["error"],
            )

        log.info(
            "ingestion_completed",
            entities=len(result.get("stored_entity_ids", [])),
            clauses=len(result.get("stored_clause_ids", [])),
        )

        return DocumentUploadResponse(
            doc_id=resolved_doc_id,
            status="completed",
            entity_count=len(result.get("stored_entity_ids", [])),
            clause_count=len(result.get("stored_clause_ids", [])),
            relationship_count=len(result.get("stored_relationship_ids", [])),
            dropped_entity_count=result.get("dropped_entity_count", 0),
        )
