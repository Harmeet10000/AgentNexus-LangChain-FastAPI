"""Unified document ingestion, search, and QA endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Response, UploadFile, status
from returns.result import Failure

from app.config import get_settings
from app.features.documents.domain.errors import DocumentValidationError
from app.kernel.result import render_result
from app.shared.rate_limit import get_rate_limiter
from app.shared.upload_validation import read_bounded_upload, upload_issue_details
from app.utils import APIResponse

from .dependencies import (
    DocumentAuthorizationDep,
    DocumentCommandServiceDep,
    DocumentQueryServiceDep,
)
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

limit_document_upload = get_rate_limiter(3, 3, 300, scope="document-upload")
limit_document_search = get_rate_limiter(20, 20, 60, scope="document-search")
limit_document_qa = get_rate_limiter(5, 5, 60, scope="document-qa")


def _narrow_document_scope(
    authorization: DocumentAuthorizationDep,
    requested: list[str],
    *,
    explicit: bool,
) -> tuple[str, ...]:
    """Client filters can narrow authority; explicit empty means deny-all."""
    if explicit and not requested:
        return ()
    return authorization.narrow_document_ids(requested)


def _scope_filter_was_provided(model: object, field: str) -> bool:
    fields = getattr(model, "model_fields_set", set())
    return field in fields


@router.post(
    path="/documents/upload",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(limit_document_upload)],
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    service: DocumentCommandServiceDep,
    authorization: DocumentAuthorizationDep,
    response: Response,
) -> APIResponse[DocumentUploadResponse]:
    settings = get_settings()
    upload_result = await read_bounded_upload(
        file,
        max_size=settings.MAX_UPLOAD_SIZE,
        allowed_extensions=settings.ALLOWED_EXTENSIONS,
    )
    if isinstance(upload_result, Failure):
        issue = upload_result.failure()
        return render_result(
            Failure(
                DocumentValidationError(
                    message=issue.message,
                    details=upload_issue_details(issue),
                    source="documents_router",
                )
            ),
            response,
            message="Document queued for ingestion",
            success_status=status.HTTP_201_CREATED,
        )
    raw_bytes = upload_result.unwrap()
    content_type = file.content_type or "application/octet-stream"
    result = await service.upload_document(
        user_id=authorization.user_id,
        tenant_id=authorization.tenant_id,
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
    authorization: DocumentAuthorizationDep,
    response: Response,
) -> APIResponse[DocumentStatusResponse]:
    result = await service.get_status(
        user_id=authorization.user_id,
        tenant_id=authorization.tenant_id,
        document_id=doc_id,
    )
    return render_result(result, response, message="Document ingestion status")


@router.post(
    "/search",
    dependencies=[Depends(limit_document_search)],
)
async def unified_search(
    payload: UnifiedSearchRequest,
    service: DocumentQueryServiceDep,
    authorization: DocumentAuthorizationDep,
    response: Response,
) -> APIResponse[UnifiedSearchResponse]:
    result = await service.search(
        user_id=authorization.user_id,
        tenant_id=authorization.tenant_id,
        allowed_doc_ids=_narrow_document_scope(
            authorization,
            payload.metadata_filter.document_ids,
            explicit=_scope_filter_was_provided(payload.metadata_filter, "document_ids"),
        ),
        payload=payload,
    )
    return render_result(result, response, message="Unified search results")


@router.post(
    path="/search/rag",
    dependencies=[Depends(limit_document_qa)],
)
async def unified_rag(
    payload: UnifiedRagRequest,
    service: DocumentQueryServiceDep,
    authorization: DocumentAuthorizationDep,
    response: Response,
) -> APIResponse[UnifiedRagResponse]:
    result = await service.rag(
        user_id=authorization.user_id,
        tenant_id=authorization.tenant_id,
        allowed_doc_ids=_narrow_document_scope(
            authorization,
            payload.metadata_filter.document_ids,
            explicit=_scope_filter_was_provided(payload.metadata_filter, "document_ids"),
        ),
        payload=payload,
    )
    return render_result(result, response, message="Unified RAG results")


@router.post(
    path="/search/ask",
    dependencies=[Depends(limit_document_qa)],
)
async def ask_corpus(
    payload: UnifiedAskRequest,
    service: DocumentQueryServiceDep,
    authorization: DocumentAuthorizationDep,
    response: Response,
) -> APIResponse[UnifiedAskResponse]:
    result = await service.ask(
        user_id=authorization.user_id,
        tenant_id=authorization.tenant_id,
        allowed_doc_ids=_narrow_document_scope(
            authorization,
            payload.doc_ids_filter,
            explicit=_scope_filter_was_provided(payload, "doc_ids_filter"),
        ),
        payload=payload,
        require_graphiti_verified=False,
    )
    return render_result(result, response, message="Grounded corpus answer")


@router.post(
    path="/legal/ask",
    dependencies=[Depends(limit_document_qa)],
)
async def ask_legal(
    payload: UnifiedAskRequest,
    service: DocumentQueryServiceDep,
    authorization: DocumentAuthorizationDep,
    response: Response,
) -> APIResponse[UnifiedAskResponse]:
    result = await service.ask(
        user_id=authorization.user_id,
        tenant_id=authorization.tenant_id,
        allowed_doc_ids=_narrow_document_scope(
            authorization,
            payload.doc_ids_filter,
            explicit=_scope_filter_was_provided(payload, "doc_ids_filter"),
        ),
        payload=payload,
        require_graphiti_verified=True,
    )
    return render_result(result, response, message="Grounded legal answer")
