"""
Ingestion router: POST /ingestion/documents/upload

Accepts multipart form: file + metadata.
Passes uploaded bytes to IngestionGraph; the graph performs Docling parsing.

Dependencies read from app.state — same pattern as agent_saul.
"""

from fastapi import APIRouter, Response, UploadFile, status

from app.shared.result import render_result
from app.utils import APIResponse, logger

from .dependencies import IngestionGraphDep, UserIdDep
from .dto import DocumentUploadResponse
from .service import IngestionService

# Concrete response type for OpenAPI schema generation
IngestionUploadResponse = APIResponse[DocumentUploadResponse]
IngestionUploadResponse.model_rebuild()


router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/documents/upload",
    summary="Upload a legal document for ingestion. Returns doc_id for use with the Agent Saul WS.",
)
async def upload_document(
    file: UploadFile,
    graph: IngestionGraphDep,
    user_id: UserIdDep,
    response: Response,
    *,
    document_type: str = "unknown",
    jurisdiction: str = "India",
) -> IngestionUploadResponse:
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
    result = await service.ingest_document(
        raw_bytes=raw_bytes,
        user_id=user_id,
        filename=file.filename or "uploaded-document",
        source=file.filename or "uploaded-document",
        document_type=document_type,
        jurisdiction=jurisdiction,
    )

    return render_result(
        result,
        response,
        message="Document Ingested Successfully",
        success_status=status.HTTP_201_CREATED,
    )
