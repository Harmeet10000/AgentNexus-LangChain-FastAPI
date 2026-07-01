"""Unified document ingestion, search, and QA endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, Path, UploadFile, status

from app.utils import APIResponse, ValidationException, http_response

from . import dependencies as documents_dependencies
from . import dto as documents_dto
from .dto import (
    DocumentStatusResponse,
    DocumentUploadResponse,
    UnifiedAskResponse,
    UnifiedRagResponse,
    UnifiedSearchResponse,
)

router = APIRouter(tags=["documents"])


@router.post(
    "/documents/upload",
    response_model=APIResponse[DocumentUploadResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    service: documents_dependencies.DocumentCommandServiceDep,
    user_id: documents_dependencies.UserIdDep,
) -> APIResponse[DocumentUploadResponse]:
    content_type = file.content_type or "application/octet-stream"
    raw_bytes = await file.read()
    if not file.filename:
        message = "Filename is required"
        raise ValidationException(message)
    response = await service.upload_document(
        user_id=user_id,
        filename=file.filename,
        content_type=content_type,
        raw_bytes=raw_bytes,
    )
    return http_response(
        "Document queued for ingestion", data=response, status_code=status.HTTP_201_CREATED
    )


@router.get(
    "/documents/{doc_id}/status",
    response_model=APIResponse[DocumentStatusResponse],
)
async def get_document_status(
    doc_id: Annotated[str, Path(min_length=1)],
    service: documents_dependencies.DocumentCommandServiceDep,
    user_id: documents_dependencies.UserIdDep,
) -> APIResponse[DocumentStatusResponse]:
    response: DocumentStatusResponse = await service.get_status(user_id=user_id, document_id=doc_id)
    return http_response("Document ingestion status", data=response)


@router.post(
    "/search",
    response_model=APIResponse[UnifiedSearchResponse],
)
async def unified_search(
    payload: documents_dto.UnifiedSearchRequest,
    service: documents_dependencies.DocumentQueryServiceDep,
    user_id: documents_dependencies.UserIdDep,
) -> APIResponse[UnifiedSearchResponse]:
    response: UnifiedSearchResponse = await service.search(user_id=user_id, payload=payload)
    return http_response("Unified search results", data=response)


@router.post(
    "/search/rag",
    response_model=APIResponse[UnifiedRagResponse],
)
async def unified_rag(
    payload: documents_dto.UnifiedRagRequest,
    service: documents_dependencies.DocumentQueryServiceDep,
    user_id: documents_dependencies.UserIdDep,
) -> APIResponse[UnifiedRagResponse]:
    response: UnifiedRagResponse = await service.rag(user_id=user_id, payload=payload)
    return http_response("Unified RAG results", data=response)


@router.post(
    "/search/ask",
    response_model=APIResponse[UnifiedAskResponse],
)
async def ask_corpus(
    payload: documents_dto.UnifiedAskRequest,
    service: documents_dependencies.DocumentQueryServiceDep,
    user_id: documents_dependencies.UserIdDep,
) -> APIResponse[UnifiedAskResponse]:
    response: UnifiedAskResponse = await service.ask(
        user_id=user_id, payload=payload, require_graphiti_verified=False
    )
    return http_response("Grounded corpus answer", data=response)


@router.post(
    "/legal/ask",
    response_model=APIResponse[UnifiedAskResponse],
)
async def ask_legal(
    payload: documents_dto.UnifiedAskRequest,
    service: documents_dependencies.DocumentQueryServiceDep,
    user_id: documents_dependencies.UserIdDep,
) -> APIResponse[UnifiedAskResponse]:
    response: UnifiedAskResponse = await service.ask(
        user_id=user_id, payload=payload, require_graphiti_verified=True
    )
    return http_response("Grounded legal answer", data=response)
