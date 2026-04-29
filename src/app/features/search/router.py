"""Search feature API endpoints."""

from typing import Annotated

from fastapi import APIRouter, Path, status

from app.utils import APIResponse, http_response

from .dependencies import SearchServiceDep, UserIdDep
from .dto import (
    HybridSearchRequest,
    LegalAskRequest,
    LegalAskResponse,
    RagSearchRequest,
    RagSearchResponse,
    SearchIngestRequest,
    SearchIngestResponse,
    SearchResponse,
    SearchTaskStatusResponse,
)

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    payload: SearchIngestRequest,
    service: SearchServiceDep,
) -> APIResponse[SearchIngestResponse]:
    response: SearchIngestResponse = await service.ingest_document(payload)
    return http_response(
        "Search document queued", data=response, status_code=status.HTTP_201_CREATED
    )


@router.get(
    "/ingest/{task_id}",
)
async def get_ingest_status(
    task_id: Annotated[str, Path(min_length=1)],
    service: SearchServiceDep,
) -> APIResponse[SearchTaskStatusResponse]:
    response: SearchTaskStatusResponse = await service.get_ingest_status(task_id)
    return http_response("Search ingest status", data=response)


@router.post(
    "/hybrid",
)
async def hybrid_search(
    payload: HybridSearchRequest,
    service: SearchServiceDep,
) -> APIResponse[SearchResponse]:
    response: SearchResponse = await service.hybrid_search(payload)
    return http_response("Hybrid search results", data=response)


@router.post(
    "/rag",
)
async def rag_search(
    payload: RagSearchRequest,
    service: SearchServiceDep,
) -> APIResponse[RagSearchResponse]:
    response: RagSearchResponse = await service.rag_search(payload)
    return http_response("RAG search results", data=response)


@router.post(
    "/ask",
)
async def ask_legal_kb(
    payload: LegalAskRequest,
    service: SearchServiceDep,
    user_id: UserIdDep,
) -> APIResponse[LegalAskResponse]:
    response: LegalAskResponse = await service.ask_legal(payload, user_id=user_id)
    return http_response("Grounded legal answer", data=response)
