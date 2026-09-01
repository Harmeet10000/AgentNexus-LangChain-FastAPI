"""Unified document ingestion, search, and QA endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, Path, Response, UploadFile, status
from returns.result import Failure

from app.shared.result import render_result
from app.utils import APIResponse

from .dependencies import DocumentCommandServiceDep, DocumentQueryServiceDep, UserIdDep
from .dto import (
    DocumentStatusResponse,
    DocumentUploadResponse,
    UnifiedAskRequest,
    UnifiedAskResponse,
    UnifiedRagRequest,
    UnifiedRagResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from .errors import DocumentValidationError

router = APIRouter(tags=["documents"])


@router.post(
    path="/documents/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    service: DocumentCommandServiceDep,
    user_id: UserIdDep,
    response: Response,
) -> APIResponse[DocumentUploadResponse]:
    content_type = file.content_type or "application/octet-stream"
    raw_bytes = await file.read()
    if not file.filename:
        return render_result(
            Failure(
                DocumentValidationError(message="Filename is required", source="documents_router")
            ),
            response,
            message="Document queued for ingestion",
            success_status=status.HTTP_201_CREATED,
        )
    result = await service.upload_document(
        user_id=user_id,
        filename=file.filename,
        content_type=content_type,
        raw_bytes=raw_bytes,
    )
    return render_result(
        result,
        response,
        message="Document queued for ingestion",
        success_status=status.HTTP_201_CREATED,
    )


@router.get(
    "/documents/{doc_id}/status",
)
async def get_document_status(
    doc_id: Annotated[str, Path(min_length=1)],
    service: DocumentCommandServiceDep,
    user_id: UserIdDep,
    response: Response,
) -> APIResponse[DocumentStatusResponse]:
    result = await service.get_status(user_id=user_id, document_id=doc_id)
    return render_result(result, response, message="Document ingestion status")


@router.post(
    "/search",
)
async def unified_search(
    payload: UnifiedSearchRequest,
    service: DocumentQueryServiceDep,
    user_id: UserIdDep,
    response: Response,
) -> APIResponse[UnifiedSearchResponse]:
    result = await service.search(user_id=user_id, payload=payload)
    return render_result(result, response, message="Unified search results")


@router.post(
    path="/search/rag",
)
async def unified_rag(
    payload: UnifiedRagRequest,
    service: DocumentQueryServiceDep,
    user_id: UserIdDep,
    response: Response,
) -> APIResponse[UnifiedRagResponse]:
    result = await service.rag(user_id=user_id, payload=payload)
    return render_result(result, response, message="Unified RAG results")


@router.post(
    path="/search/ask",
)
async def ask_corpus(
    payload: UnifiedAskRequest,
    service: DocumentQueryServiceDep,
    user_id: UserIdDep,
    response: Response,
) -> APIResponse[UnifiedAskResponse]:
    result = await service.ask(user_id=user_id, payload=payload, require_graphiti_verified=False)
    return render_result(result, response, message="Grounded corpus answer")


@router.post(
    path="/legal/ask",
)
async def ask_legal(
    payload: UnifiedAskRequest,
    service: DocumentQueryServiceDep,
    user_id: UserIdDep,
    response: Response,
) -> APIResponse[UnifiedAskResponse]:
    result = await service.ask(user_id=user_id, payload=payload, require_graphiti_verified=True)
    return render_result(result, response, message="Grounded legal answer")
