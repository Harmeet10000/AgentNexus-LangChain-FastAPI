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
    "AutocompleteResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "SearchService",
    "get_search_repository",
    "get_search_service",
    "router",
]
