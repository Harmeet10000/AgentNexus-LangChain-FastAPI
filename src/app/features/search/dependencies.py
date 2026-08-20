"""Dependency wiring for the search feature."""

from typing import Annotated

from fastapi import Depends, Request
from langchain_core.language_models import BaseChatModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db, get_redis
from app.features.auth import CurrentClaims
from app.shared.langchain_layer.models import _build_chat_model

from .repository import SearchRepository
from .service import SearchService


async def get_search_repository(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> SearchRepository:
    return SearchRepository(session)


def _get_search_llm() -> BaseChatModel:
    return _build_chat_model(
        model_name=None,
        temperature=0.1,
        implementation="generic",
    )


async def get_search_service(
    repo: Annotated[SearchRepository, Depends(get_search_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
    request: Request,
) -> SearchService:
    return SearchService(
        repo=repo,
        llm=_get_search_llm(),
        redis=redis,
        graphiti=getattr(request.app.state, "graphiti", None),
    )


async def get_current_user_id(claims: CurrentClaims) -> str:
    return claims.sub


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
UserIdDep = Annotated[str, Depends(get_current_user_id)]
