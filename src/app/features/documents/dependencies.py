"""Dependency wiring for unified document feature."""

from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.connections import get_postgres_db, get_redis
from app.shared.services.storage import StorageService

from .repository import DocumentRepository
from .service import DocumentCommandService, DocumentQueryService


async def get_document_repository(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> DocumentRepository:
    return DocumentRepository(session)


async def get_document_command_service(
    repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    request: Request,
) -> DocumentCommandService:
    object_store = getattr(request.app.state, "object_store", None)
    if object_store is None:
        object_store = StorageService.from_settings(settings=get_settings())
    return DocumentCommandService(repo=repo, object_store=object_store)


async def get_document_query_service(
    repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
    request: Request,
) -> DocumentQueryService:
    return DocumentQueryService(
        repo=repo,
        redis=redis,
        graphiti=getattr(request.app.state, "graphiti", None),
    )


async def get_current_user_id(request: Request) -> str:
    return request.state.user_id


DocumentCommandServiceDep = Annotated[DocumentCommandService, Depends(get_document_command_service)]
DocumentQueryServiceDep = Annotated[DocumentQueryService, Depends(get_document_query_service)]
UserIdDep = Annotated[str, Depends(get_current_user_id)]
