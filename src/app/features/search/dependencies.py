"""Dependency wiring for the search feature."""

from typing import Annotated

from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db, get_redis

from .repository import SearchRepository
from .service import SearchService


async def get_search_repository(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> SearchRepository:
    return SearchRepository(session)


async def get_search_service(
    repo: Annotated[SearchRepository, Depends(get_search_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> SearchService:
    return SearchService(repo=repo, redis=redis)


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
