from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db, get_redis
from app.features.search.repository import SearchRepository
from app.features.search.service import SearchService


async def get_search_repository(
    session: AsyncSession = Depends(get_postgres_db),
) -> SearchRepository:
    return SearchRepository(session)


async def get_search_service(
    repo=Depends(get_search_repository),
    redis=Depends(get_redis),
) -> SearchService:
    return SearchService(repo, redis=redis)
