"""
Ingestion router: POST /ingestion/documents/upload

Accepts multipart form: file + metadata.
Passes uploaded bytes to IngestionGraph; the graph performs Docling parsing.

Dependencies read from app.state — same pattern as agent_saul.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, UploadFile

from app.utils import APIResponse, http_response, logger

from .dto import DocumentUploadResponse
from .service import IngestionService

if TYPE_CHECKING:
    from .dependencies import IngestionGraphDep, UserIdDep

router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/documents/upload",
    response_model=APIResponse[DocumentUploadResponse],
    summary="Upload a legal document for ingestion. Returns doc_id for use with the Agent Saul WS.",
)
async def upload_document(
    file: UploadFile,
    graph: IngestionGraphDep,
    user_id: UserIdDep,
    document_type: str = "unknown",
    jurisdiction: str = "India",
) -> APIResponse[DocumentUploadResponse]:
    """
    Upload flow:
      1. Read raw bytes from uploaded file.
      2. Run IngestionGraph (parse → schema → segment → contextualize → store).
      4. Return doc_id for use in WS /agent-saul/ws/{thread_id}.
    """
    raw_bytes = await file.read()

    log = logger.bind(user_id=user_id, filename=file.filename, doc_type=document_type)
    log.info("upload_received", size_bytes=len(raw_bytes))

    service = IngestionService(ingestion_graph=graph)
    response: DocumentUploadResponse = await service.ingest_document(
        raw_bytes=raw_bytes,
        user_id=user_id,
        filename=file.filename or "uploaded-document",
        source=file.filename or "uploaded-document",
        document_type=document_type,
        jurisdiction=jurisdiction,
    )

    log.info("upload_processed", doc_id=response.doc_id, status=response.status)
    return http_response(
        message="Document Ingested Successfully",
        data=response,
        status_code=201,
    )
