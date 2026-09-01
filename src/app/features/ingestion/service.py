"""
IngestionService: runs IngestionGraph for a given uploaded document.

Called by the HTTP router. Non-streaming — awaits completion before returning.
If IngestionGraph returns an expected typed failure, maps it to the existing
project exception boundary before returning to FastAPI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph
from returns.result import Failure, Success

from app.shared.result import FeatureError
from app.utils import logger

from .dto import DocumentUploadResponse
from .errors import (
    IngestionGraphError,
    IngestionInternalError,
    IngestionPipelineError,
)

if TYPE_CHECKING:
    from typing import Any

    from .errors import IngestionResult


class IngestionService:
    def __init__(self, ingestion_graph: CompiledStateGraph[Any]) -> None:
        self._graph = ingestion_graph

    async def ingest_document(
        self,
        raw_bytes: bytes,
        user_id: str,
        filename: str,
        source: str,
        *,
        document_type: str = "unknown",
        jurisdiction: str = "India",
        doc_id: str | None = None,
    ) -> IngestionResult[DocumentUploadResponse]:
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
            return Failure(
                IngestionGraphError(
                    message="Document ingestion failed",
                    details={"doc_id": resolved_doc_id, "error": str(exc)},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            )

        failure = result.get("failure")
        if failure is not None:
            if isinstance(failure, FeatureError):
                error = IngestionPipelineError(
                    message=failure.message,
                    details=failure.details or {"doc_id": resolved_doc_id},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            elif isinstance(failure, dict):
                error = IngestionPipelineError(
                    message=str(failure.get("message", "Document ingestion failed")),
                    details={"doc_id": resolved_doc_id, "failure": failure},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            else:
                error = IngestionInternalError(
                    message=str(failure),
                    details={"doc_id": resolved_doc_id, "failure": str(failure)},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            return Failure(error)

        log.info(
            "ingestion_completed",
            entities=len(result.get("stored_entity_ids", [])),
            clauses=len(result.get("stored_clause_ids", [])),
        )

        return Success(
            DocumentUploadResponse(
                doc_id=resolved_doc_id,
                status="completed",
                entity_count=len(result.get("stored_entity_ids", [])),
                clause_count=len(result.get("stored_clause_ids", [])),
                relationship_count=len(result.get("stored_relationship_ids", [])),
                dropped_entity_count=result.get("dropped_entity_count", 0),
            )
        )
