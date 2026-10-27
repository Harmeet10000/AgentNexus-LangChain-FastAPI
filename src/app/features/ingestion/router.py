"""
Legacy ingestion router: POST /ingestion/documents/upload.

Accepts multipart form: file + metadata.
The endpoint remains available for compatibility, but new clients should use
``POST /documents/upload``. Both paths use the canonical document-ingestion graph;
this route is retained only while existing clients migrate.

Dependencies read from app.state — same pattern as agent_saul.
"""

from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Depends, Response, UploadFile, status
from returns.result import Failure

from app.config import get_settings
from app.features.auth.dependencies import AuthorizationContext
from app.features.documents.dependencies import DocumentAuthorizationDep, DocumentCommandServiceDep
from app.kernel.result import render_result
from app.observability.logging import logger
from app.resilience.rate_limit import get_rate_limiter
from app.shared.upload_validation import read_bounded_upload, upload_issue_details
from app.utils import APIResponse

from .dto import DocumentUploadResponse
from .exceptions import IngestionValidationError
from .service import IngestionService

if TYPE_CHECKING:
    from typing import Any

# Concrete response type for OpenAPI schema generation
IngestionUploadResponse = APIResponse[DocumentUploadResponse]
IngestionUploadResponse.model_rebuild()


router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
)

_LEGACY_SUCCESSOR_PATH = "/documents/upload"


def _mark_legacy_upload(response: Response) -> None:
    """Advertise the canonical upload route on every legacy response."""
    settings = get_settings()
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = settings.API_SUNSET_DATE
    response.headers["Link"] = (
        f'<{_LEGACY_SUCCESSOR_PATH}>; rel="successor-version"; title="Canonical document upload"'
    )


def _log_legacy_usage(*, user_id: str, filename: str | None, sunset: str) -> None:
    """Emit a structured migration signal for every legacy-route request.

    The route is retained because production consumers are not currently known.
    A stable event name and migration fields let operators discover those consumers
    from logs before the compatibility window expires, without making the request
    depend on a separate metrics or telemetry service.
    """
    logger.bind(
        event="legacy_ingestion_upload_deprecated",
        endpoint="/ingestion/documents/upload",
        successor_endpoint=_LEGACY_SUCCESSOR_PATH,
        user_id=user_id,
        filename=filename,
        sunset=sunset,
        migration_action="migrate_to_documents_upload",
    ).info("Deprecated ingestion upload endpoint used")


limit_document_upload = get_rate_limiter(3, 3, 300, scope="document-upload")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/documents/upload",
    summary="Deprecated: upload a legal document",
    description=(
        "Deprecated compatibility endpoint. Use POST /documents/upload instead. "
        "The response includes Deprecation, Sunset, and successor-version Link headers."
    ),
    deprecated=True,
    dependencies=[Depends(limit_document_upload)],
)
async def upload_document(
    file: UploadFile,
    service: DocumentCommandServiceDep,
    authorization: DocumentAuthorizationDep,
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
    _mark_legacy_upload(response)
    settings = get_settings()
    user_id = (
        authorization.user_id
        if isinstance(authorization, AuthorizationContext)
        else str(authorization)
    )
    _log_legacy_usage(user_id=user_id, filename=file.filename, sunset=settings.API_SUNSET_DATE)
    upload_result = await read_bounded_upload(
        file,
        max_size=settings.MAX_UPLOAD_SIZE,
        allowed_extensions=settings.ALLOWED_EXTENSIONS,
    )
    if isinstance(upload_result, Failure):
        issue = upload_result.failure()
        return render_result(
            Failure(
                IngestionValidationError(
                    message=issue.message,
                    details=upload_issue_details(issue),
                    source="ingestion_router",
                )
            ),
            response,
            message="Document ingestion rejected",
            success_status=status.HTTP_201_CREATED,
        )
    raw_bytes = upload_result.unwrap()

    # Production requests always receive the canonical tenant-aware command
    # service.  The tiny compatibility branch exists only for direct callers
    # of this function from pre-migration tests/clients that still inject the
    # old compiled graph; it is not a FastAPI dependency path.
    if (
        hasattr(service, "upload_document")
        and hasattr(authorization, "user_id")
        and hasattr(authorization, "tenant_id")
    ):
        result = await service.upload_document(
            user_id=authorization.user_id,
            tenant_id=authorization.tenant_id,
            filename=file.filename or "uploaded-document",
            content_type=file.content_type or "application/octet-stream",
            raw_bytes=raw_bytes,
        )
        return render_result(
            result,
            response,
            message="Document queued for ingestion",
            success_status=status.HTTP_201_CREATED,
        )

    user_id = str(authorization)
    graph = service
    log = logger.bind(
        user_id=user_id,
        filename=file.filename,
        doc_type=document_type,
        legacy_endpoint=True,
        migration_target=_LEGACY_SUCCESSOR_PATH,
    )
    log.info("upload_received", size_bytes=len(raw_bytes))

    legacy_service = IngestionService(ingestion_graph=cast("Any", graph))
    result = await legacy_service.ingest_document(
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
