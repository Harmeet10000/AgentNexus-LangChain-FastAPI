"""Search feature API endpoints."""

from fastapi import APIRouter, Depends, status

from app.features.search.dependencies import get_search_service
from app.features.search.dto import (
    AutocompleteResponse,
    DocumentVectorCreate,
    DocumentVectorResponse,
    SearchRequest,
    SearchResponse,
)
from app.features.search.service import SearchService

router = APIRouter(prefix="/searches", tags=["Searches"])


@router.post(path="", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Search documents using hybrid search (semantic + keyword)."""
    return await service.perform_search(request)


@router.get(path="/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(
    query: str,
    service: SearchService = Depends(get_search_service),
) -> AutocompleteResponse:
    """Get search suggestions based on partial query."""
    return await service.get_suggestions(query)


@router.post(
    path="/documents",
    response_model=DocumentVectorResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    data: DocumentVectorCreate,
    service: SearchService = Depends(get_search_service),
) -> DocumentVectorResponse:
    """Insert a new document vector into the database."""
    return await service.create_document(data)
