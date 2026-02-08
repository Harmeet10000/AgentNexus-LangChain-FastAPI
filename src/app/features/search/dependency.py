from fastapi import Depends

from app.connections.mongodb import get_db
from app.connections.redis import get_redis
from app.features.search.repository import SearchRepository
from app.features.search.service import SearchService


def get_search_repository(db=Depends(get_db)) -> SearchRepository:
    return SearchRepository(db)


def get_search_service(
    repo=Depends(get_search_repository),
    redis=Depends(get_redis),
) -> SearchService:
    return SearchService(repo, redis)
