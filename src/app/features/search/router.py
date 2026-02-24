"""Search feature API endpoints."""

from fastapi import APIRouter, Depends

from app.features.search.dependencies import get_search_service
from app.features.search.dto import AutocompleteResponse, SearchRequest, SearchResponse
from app.features.search.service import SearchService

router = APIRouter(prefix="/searches", tags=["Searches"])


@router.post("", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search documents using hybrid search (semantic + keyword)."""
    return await service.perform_search(request)


@router.get("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(
    query: str,
    service: SearchService = Depends(get_search_service),
) -> AutocompleteResponse:
    """Get search suggestions based on partial query."""
    return await service.get_suggestions(query)
