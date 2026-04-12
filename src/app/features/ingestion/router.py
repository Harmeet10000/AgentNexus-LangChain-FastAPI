"""
Ingestion router: POST /ingestion/documents/upload

Accepts multipart form: file + metadata.
Extracts text (stub — wire Docling here), runs IngestionGraph, returns doc_id.

Dependencies read from app.state — same pattern as agent_saul.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, UploadFile

from app.shared import APIResponse
from app.utils import http_response, logger

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
      2. Extract text (Docling stub — replace with real Docling call).
      3. Run IngestionGraph (extract → validate → embed_store).
      4. Return doc_id for use in WS /agent-saul/ws/{thread_id}.
    """
    raw_bytes = await file.read()

    # TODO: replace with real Docling extraction
    # from docling import DocumentConverter
    # converter = DocumentConverter()
    # result = converter.convert_from_bytes(raw_bytes)
    # raw_text = result.document.export_to_text()
    raw_text = raw_bytes.decode("utf-8", errors="replace")  # stub

    log = logger.bind(user_id=user_id, filename=file.filename, doc_type=document_type)
    log.info("upload_received", size_bytes=len(raw_bytes))

    service = IngestionService(ingestion_graph=graph)
    response = await service.ingest_document(
        raw_text=raw_text,
        user_id=user_id,
        document_type=document_type,
        jurisdiction=jurisdiction,
    )

    log.info("upload_processed", doc_id=response.doc_id, status=response.status)
    return http_response(
        message="Document Ingested Successfully",
        data=response,
        status_code=201,
    )
