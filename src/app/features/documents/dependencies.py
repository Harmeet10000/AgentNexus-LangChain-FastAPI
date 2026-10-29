"""Dependency wiring for unified document feature."""

from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import Depends, Request
from langchain_core.language_models import BaseChatModel
from redis.asyncio import Redis
from returns.result import Failure
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.connections import get_postgres_db, get_redis
from app.features.auth.api.http.dependencies import (
    AuthorizationContext,
    CurrentAuthorizationContext,
)
from app.shared.langchain_layer import build_chat_model
from app.utils import InfrastructureException
from app.utils.storage import StorageService

from .repository import DocumentRepository
from .service import DocumentCommandService, DocumentQueryService

if TYPE_CHECKING:
    from app.ai.generation import TextGenerator
    from app.config.settings import Settings

    from .access_audit import DocumentAccessAuditRecorder


async def get_document_repository(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> DocumentRepository:
    return DocumentRepository(session)


def _get_document_llm() -> BaseChatModel:
    # Deliberately NOT called at dependency-resolution time. Constructing the
    # model provider client while the caller is still unauthenticated lets its
    # failure (missing package, bad key) surface as a 500 that masks the 401
    # the request had already earned — FastAPI resolves a path operation's
    # dependencies as a set and answers with whichever raised first.
    return build_chat_model(
        model_name=None,
        temperature=0.1,
        implementation="generic",
    )


async def get_document_command_service(
    repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    request: Request,
) -> DocumentCommandService:
    object_store: StorageService | None = getattr(request.app.state, "object_store", None)
    if object_store is None:
        settings: Settings = get_settings()
        if settings.S3_BUCKET_NAME:
            object_store: StorageService = StorageService.from_settings(settings=settings)
    recorder = getattr(getattr(request.app.state, "evidence_service", None), "audit_recorder", None)
    return DocumentCommandService(
        repo=repo,
        object_store=object_store,
        audit_recorder=cast("DocumentAccessAuditRecorder | None", recorder),
    )


async def get_document_query_service(
    repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    redis: Annotated[Redis, Depends(dependency=get_redis)],
    request: Request,
) -> DocumentQueryService:
    # The context grader rides the optional text-generation seam the lifespan
    # provisions. Absence (generation disabled) stays `None`: the service keeps
    # grading on its legacy inline path, so this injection never raises.
    grader = cast("TextGenerator | None", getattr(request.app.state, "text_generator", None))
    recorder = getattr(getattr(request.app.state, "evidence_service", None), "audit_recorder", None)
    return DocumentQueryService(
        repo=repo,
        llm_factory=_get_document_llm,
        redis=redis,
        graphiti=getattr(request.app.state, "graphiti", None),
        grader=grader,
        audit_recorder=cast("DocumentAccessAuditRecorder | None", recorder),
    )


async def get_current_user_id(context: CurrentAuthorizationContext) -> str:
    return context.user_id


async def get_document_authorization(
    context: CurrentAuthorizationContext,
    repo: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> AuthorizationContext:
    """Expose a complete server-derived capability to document handlers.

    The request's document IDs are only filters.  The capability itself comes
    from a tenant-scoped SQL query, so forged IDs can never widen authority.
    """
    result = await repo.list_document_ids_for_tenant(tenant_id=context.tenant_id)
    if isinstance(result, Failure):
        message = "Unable to resolve tenant document capability"
        raise InfrastructureException(message)
    thread_id = context.thread_id or f"http:{context.session_id or context.user_id}"
    return replace(context.for_thread(thread_id), allowed_doc_ids=result.unwrap())


DocumentCommandServiceDep = Annotated[DocumentCommandService, Depends(get_document_command_service)]
DocumentQueryServiceDep = Annotated[DocumentQueryService, Depends(get_document_query_service)]
UserIdDep = Annotated[str, Depends(get_current_user_id)]
DocumentAuthorizationDep = Annotated[AuthorizationContext, Depends(get_document_authorization)]
