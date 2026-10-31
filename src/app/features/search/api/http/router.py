"""Site-search API endpoints.

All endpoints require authentication (bearer token + live tenant membership),
matching ``features/documents/api/http/router.py``. Expected failures arrive as
``Result`` and are rendered by ``render_result``; nothing on the HTTP path
raises a typed feature error.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response, status

from app.kernel.result import render_result
from app.shared.rate_limit import get_rate_limiter
from app.utils import APIResponse

from .dependencies import SearchAuthorizationDep, SearchServiceDep
from .dto import (
    HybridSearchRequest,
    RagSearchRequest,
    RagSearchResponse,
    SearchIngestRequest,
    SearchIngestResponse,
    SearchResponse,
    SearchTaskStatusResponse,
)

router = APIRouter(prefix="/search", tags=["search"])

limit_search = get_rate_limiter(20, 20, 60, scope="search")


@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(limit_search)],
)
async def ingest_document(
    payload: SearchIngestRequest,
    service: SearchServiceDep,
    authorization: SearchAuthorizationDep,
    response: Response,
) -> APIResponse[SearchIngestResponse]:
    result = await service.ingest_document(tenant_id=authorization.tenant_id, payload=payload)
    return render_result(
        result,
        response,
        message="Search document queued",
        success_status=status.HTTP_201_CREATED,
    )


@router.get(
    "/ingest/{task_id}",
    dependencies=[Depends(limit_search)],
)
async def get_ingest_status(
    task_id: Annotated[str, Path(min_length=1)],
    service: SearchServiceDep,
    authorization: SearchAuthorizationDep,
    response: Response,
) -> APIResponse[SearchTaskStatusResponse]:
    result = await service.get_ingest_status(tenant_id=authorization.tenant_id, task_id=task_id)
    return render_result(result, response, message="Search ingest status")


@router.post(
    "/hybrid",
    dependencies=[Depends(limit_search)],
)
async def hybrid_search(
    payload: HybridSearchRequest,
    service: SearchServiceDep,
    authorization: SearchAuthorizationDep,
    response: Response,
) -> APIResponse[SearchResponse]:
    result = await service.hybrid_search(tenant_id=authorization.tenant_id, payload=payload)
    return render_result(result, response, message="Hybrid search results")


@router.post(
    "/rag",
    dependencies=[Depends(limit_search)],
)
async def rag_search(
    payload: RagSearchRequest,
    service: SearchServiceDep,
    authorization: SearchAuthorizationDep,
    response: Response,
) -> APIResponse[RagSearchResponse]:
    result = await service.rag_search(tenant_id=authorization.tenant_id, payload=payload)
    return render_result(result, response, message="RAG search results")
