"""
Ingestion router: POST /ingestion/documents/upload

Accepts multipart form: file + metadata.
Extracts text (stub — wire Docling here), runs IngestionGraph, returns doc_id.

Dependencies read from app.state — same pattern as agent_saul.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph

from src.app.features.ingestion.dto import DocumentUploadResponse
from src.app.features.ingestion.service import IngestionService
from src.app.shared.response_type import APIResponse, http_response

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def get_ingestion_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.ingestion_graph  # type: ignore[no-any-return]


async def get_current_user_id(request: Request) -> str:
    return request.state.user_id  # type: ignore[no-any-return]


IngestionGraphDep = Annotated[CompiledStateGraph, Depends(get_ingestion_graph)]
UserIdDep = Annotated[str, Depends(get_current_user_id)]


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
) -> JSONResponse:
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
    return http_response(response)
