"""Dependency wiring for unified document feature."""

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request
from langchain_core.language_models import BaseChatModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.connections import get_postgres_db, get_redis
from app.shared.langchain_layer import _build_chat_model
from app.shared.services.storage import StorageService

from .repository import DocumentRepository
from .service import DocumentCommandService, DocumentQueryService

if TYPE_CHECKING:
    from app.config.settings import Settings


async def get_document_repository(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> DocumentRepository:
    return DocumentRepository(session)


def _get_document_llm() -> BaseChatModel:
    return _build_chat_model(
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
    return DocumentCommandService(repo=repo, object_store=object_store)


async def get_document_query_service(
    repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    redis: Annotated[Redis, Depends(dependency=get_redis)],
    request: Request,
) -> DocumentQueryService:
    return DocumentQueryService(
        repo=repo,
        llm=_get_document_llm(),
        redis=redis,
        graphiti=getattr(request.app.state, "graphiti", None),
    )


async def get_current_user_id(request: Request) -> str:
    return request.state.user_id


DocumentCommandServiceDep = Annotated[DocumentCommandService, Depends(get_document_command_service)]
DocumentQueryServiceDep = Annotated[DocumentQueryService, Depends(get_document_query_service)]
UserIdDep = Annotated[str, Depends(get_current_user_id)]
