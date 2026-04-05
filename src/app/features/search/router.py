"""Search feature API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Path, status
from fastapi.responses import Response

from app.features.search.dependencies import SearchServiceDep
from app.features.search.dto import (
    HybridSearchRequest,
    RagSearchRequest,
    RagSearchResponse,
    SearchIngestRequest,
    SearchIngestResponse,
    SearchResponse,
    SearchTaskStatusResponse,
)
from app.shared.response_type import APIResponse
from app.utils import http_response

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "/ingest",
    response_model=APIResponse[SearchIngestResponse],
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    payload: SearchIngestRequest,
    service: SearchServiceDep,
) -> Response:
    response = await service.ingest_document(payload)
    return http_response(
        "Search document queued", data=response, status_code=status.HTTP_201_CREATED
    )


@router.get(
    "/ingest/{task_id}",
    response_model=APIResponse[SearchTaskStatusResponse],
)
async def get_ingest_status(
    task_id: Annotated[str, Path(min_length=1)],
    service: SearchServiceDep,
) -> Response:
    response = await service.get_ingest_status(task_id)
    return http_response("Search ingest status", data=response)


@router.post(
    "/hybrid",
    response_model=APIResponse[SearchResponse],
)
async def hybrid_search(
    payload: HybridSearchRequest,
    service: SearchServiceDep,
) -> Response:
    response = await service.hybrid_search(payload)
    return http_response("Hybrid search results", data=response)


@router.post(
    "/rag",
    response_model=APIResponse[RagSearchResponse],
)
async def rag_search(
    payload: RagSearchRequest,
    service: SearchServiceDep,
) -> Response:
    response = await service.rag_search(payload)
    return http_response("RAG search results", data=response)
