"""Unified document ingestion, search, and QA endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, Path, UploadFile, status

from app.utils import APIResponse, ValidationException, http_response

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

router = APIRouter(tags=["documents"])


@router.post(
    path="/documents/upload",
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    service: DocumentCommandServiceDep,
    user_id: UserIdDep,
) -> APIResponse[DocumentUploadResponse]:
    content_type = file.content_type or "application/octet-stream"
    raw_bytes = await file.read()
    if not file.filename:
        message = "Filename is required"
        raise ValidationException(message)
    response: DocumentUploadResponse = await service.upload_document(
        user_id=user_id,
        filename=file.filename,
        content_type=content_type,
        raw_bytes=raw_bytes,
    )
    return http_response(
        message="Document queued for ingestion", data=response, status_code=status.HTTP_201_CREATED
    )


@router.get(
    "/documents/{doc_id}/status",
)
async def get_document_status(
    doc_id: Annotated[str, Path(min_length=1)],
    service: DocumentCommandServiceDep,
    user_id: UserIdDep,
) -> APIResponse[DocumentStatusResponse]:
    response: DocumentStatusResponse = await service.get_status(user_id=user_id, document_id=doc_id)
    return http_response(message="Document ingestion status", data=response)


@router.post(
    "/search",
)
async def unified_search(
    payload: UnifiedSearchRequest,
    service: DocumentQueryServiceDep,
    user_id: UserIdDep,
) -> APIResponse[UnifiedSearchResponse]:
    response: UnifiedSearchResponse = await service.search(user_id=user_id, payload=payload)
    return http_response(message="Unified search results", data=response)


@router.post(
    path="/search/rag",
)
async def unified_rag(
    payload: UnifiedRagRequest,
    service: DocumentQueryServiceDep,
    user_id: UserIdDep,
) -> APIResponse[UnifiedRagResponse]:
    response: UnifiedRagResponse = await service.rag(user_id=user_id, payload=payload)
    return http_response("Unified RAG results", data=response)


@router.post(
    path="/search/ask",
)
async def ask_corpus(
    payload: UnifiedAskRequest,
    service: DocumentQueryServiceDep,
    user_id: UserIdDep,
) -> APIResponse[UnifiedAskResponse]:
    response: UnifiedAskResponse = await service.ask(
        user_id=user_id, payload=payload, require_graphiti_verified=False
    )
    return http_response(message="Grounded corpus answer", data=response)


@router.post(
    path="/legal/ask",
)
async def ask_legal(
    payload: UnifiedAskRequest,
    service: DocumentQueryServiceDep,
    user_id: UserIdDep,
) -> APIResponse[UnifiedAskResponse]:
    response: UnifiedAskResponse = await service.ask(
        user_id=user_id, payload=payload, require_graphiti_verified=True
    )
    return http_response(message="Grounded legal answer", data=response)
