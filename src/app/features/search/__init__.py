"""Search feature module."""

from app.features.search.dependencies import (
    get_search_repository,
    get_search_service,
)
from app.features.search.dto import (
    AutocompleteResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from app.features.search.router import router
from app.features.search.service import SearchService

__all__ = [
    "get_search_service",
    "get_search_repository",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "AutocompleteResponse",
    "SearchService",
    "router",
]
