"""Crawler feature API endpoints."""

import asyncio
import ipaddress
from typing import Annotated
from uuid import uuid4

from celery.exceptions import CeleryError
from fastapi import APIRouter, Depends, Header, Query, Request, Response

from app.config import get_settings
from app.connections.celery_task_names import CRAWLER_CRAWL
from app.shared.result import http_response, render_result
from app.shared.services import RateLimitScope
from app.shared.services.rate_limiter import RateLimiter
from app.utils import (
    APIResponse,
    NotFoundException,
    ServiceUnavailableException,
    TooManyRequestsException,
)

from .constants import CRAWLER_PREFIX, CRAWLER_TAG
from .dependencies import get_crawl_job_store, get_crawler_service, get_rate_limiter
from .dto import (
    CrawlJob,
    CrawlJobChunkList,
    CrawlJobPageList,
    CrawlJobSearchResponse,
    CrawlRequest,
    RateLimitInfo,
    SearchRequest,
    SearchResponse,
)
from .job_store import CrawlJobStore
from .service import CrawlerService

router = APIRouter(prefix=CRAWLER_PREFIX, tags=[CRAWLER_TAG])


def get_client_identifier(request: Request) -> str:
    """Use the configured trusted-proxy chain for ownership and rate limiting."""
    forwarded = request.headers.get("X-Forwarded-For")
    trusted_proxies = get_settings().FASTAPI_GUARD_TRUSTED_PROXIES
    client_host = request.client.host if request.client else "unknown"
    if forwarded and _is_trusted_proxy(client_host, trusted_proxies):
        for candidate in reversed([part.strip() for part in forwarded.split(",") if part.strip()]):
            if _is_trusted_proxy(candidate, trusted_proxies):
                continue
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                continue
            return candidate
    return client_host


def _is_trusted_proxy(host: str, trusted_proxies: list[str]) -> bool:
    try:
        address: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(host)
    except ValueError:
        return False
    for configured in trusted_proxies:
        try:
            if address in ipaddress.ip_network(configured, strict=False):
                return True
        except ValueError:
            if host == configured:
                return True
    return False


async def _owner_job(request: Request, crawl_id: str, store: CrawlJobStore) -> CrawlJob:
    job = await store.get(crawl_id, owner=get_client_identifier(request))
    if job is None:
        resource = "Crawl job"
        raise NotFoundException(resource, crawl_id)
    return job


def _cursor(value: str) -> int:
    try:
        cursor = int(value)
    except ValueError as exc:
        message = "cursor must be a non-negative integer"
        raise ValueError(message) from exc
    if cursor < 0:
        message = "cursor must be a non-negative integer"
        raise ValueError(message)
    return cursor


@router.post(path="/crawl")
async def crawl_url(  # noqa: PLR0917
    request_data: CrawlRequest,
    request: Request,
    response: Response,
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    job_store: Annotated[CrawlJobStore, Depends(get_crawl_job_store)],
    idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
) -> APIResponse[CrawlJob]:
    """Create a durable crawl job; execution always happens in a Celery worker."""
    owner = get_client_identifier(request)
    is_allowed, rate_info = await rate_limiter.acquire(owner, RateLimitScope.CRAWL)
    if not is_allowed:
        raise TooManyRequestsException(
            detail=rate_info.get("error") or "Rate limit exceeded",
            data={"retry_after": rate_info.get("retry_after")},
        )

    crawl_id = uuid4().hex
    payload = request_data.model_dump(mode="json")
    job, created = await job_store.create(
        crawl_id=crawl_id,
        owner=owner,
        request_payload=payload,
        idempotency_key=idempotency_key,
    )
    if created:
        celery = getattr(request.app.state, "celery", None)
        if celery is None:
            message = "Celery is unavailable"
            await job_store.mark_failed(job.crawl_id, message)
            message = "Durable crawler workers are unavailable"
            raise ServiceUnavailableException(message)
        try:
            task_result = await asyncio.to_thread(
                celery.send_task,
                CRAWLER_CRAWL,
                kwargs={"crawl_id": job.crawl_id, "request_payload": payload},
                queue=get_settings().CELERY_CRAWLER_QUEUE,
                routing_key=get_settings().CELERY_CRAWLER_ROUTING_KEY,
            )
            await job_store.attach_task_id(job.crawl_id, task_result.id)
        except (CeleryError, ConnectionError, OSError, TimeoutError) as exc:
            message = "Could not enqueue crawler job"
            await job_store.mark_failed(job.crawl_id, message)
            raise ServiceUnavailableException(message) from exc
    response.status_code = 202
    return http_response(message="Crawler job accepted", data=job, status_code=202)


@router.get(path="/crawl/{crawl_id}")
async def crawl_status(
    request: Request,
    crawl_id: str,
    store: Annotated[CrawlJobStore, Depends(get_crawl_job_store)],
) -> APIResponse[CrawlJob]:
    return http_response(message="Crawler job status", data=await _owner_job(request, crawl_id, store))


@router.get(path="/crawl/{crawl_id}/pages")
async def crawl_pages(
    request: Request,
    crawl_id: str,
    store: Annotated[CrawlJobStore, Depends(get_crawl_job_store)],
    cursor: Annotated[str, Query(min_length=1, max_length=32)] = "0",
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
) -> APIResponse[CrawlJobPageList]:
    await _owner_job(request, crawl_id, store)
    result = await store.pages(
        crawl_id, owner=get_client_identifier(request), cursor=_cursor(cursor), limit=limit
    )
    if result is None:
        message = "Crawler result is not available yet"
        raise ServiceUnavailableException(message)
    return http_response(message="Crawler pages", data=result)


@router.get(path="/crawl/{crawl_id}/chunks")
async def crawl_chunks(
    request: Request,
    crawl_id: str,
    store: Annotated[CrawlJobStore, Depends(get_crawl_job_store)],
    cursor: Annotated[str, Query(min_length=1, max_length=32)] = "0",
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> APIResponse[CrawlJobChunkList]:
    await _owner_job(request, crawl_id, store)
    chunk_result = await store.chunks(
        crawl_id, owner=get_client_identifier(request), cursor=_cursor(cursor), limit=limit
    )
    if chunk_result is None:
        message = "Crawler result is not available yet"
        raise ServiceUnavailableException(message)
    items, next_cursor = chunk_result
    return http_response(
        message="Crawler chunks",
        data=CrawlJobChunkList(crawl_id=crawl_id, items=items, next_cursor=next_cursor),
    )


@router.get(path="/crawl/{crawl_id}/chunks/search")
async def crawl_search_chunks(
    request: Request,
    crawl_id: str,
    store: Annotated[CrawlJobStore, Depends(get_crawl_job_store)],
    query: Annotated[str, Query(min_length=1, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> APIResponse[CrawlJobSearchResponse]:
    await _owner_job(request, crawl_id, store)
    result = await store.search_chunks(
        crawl_id, owner=get_client_identifier(request), query=query, limit=limit
    )
    if result is None:
        message = "Crawler result is not available yet"
        raise ServiceUnavailableException(message)
    return http_response(message="Crawler chunk search", data=result)


@router.post(path="/crawl/{crawl_id}/cancel")
async def cancel_crawl(
    request: Request,
    crawl_id: str,
    store: Annotated[CrawlJobStore, Depends(get_crawl_job_store)],
) -> APIResponse[CrawlJob]:
    result = await store.request_cancel(crawl_id, owner=get_client_identifier(request))
    if result is None:
        resource = "Crawl job"
        raise NotFoundException(resource, crawl_id)
    task_id = await store.task_id(crawl_id, owner=get_client_identifier(request))
    celery = getattr(request.app.state, "celery", None)
    if task_id and celery is not None:
        await asyncio.to_thread(celery.control.revoke, task_id, terminate=False)
    return http_response(message="Crawler cancellation requested", data=result)


@router.get(path="/search")
async def search_web(
    request: Request,
    response: Response,
    service: Annotated[CrawlerService, Depends(get_crawler_service)],
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    query: Annotated[str, Query(min_length=1, max_length=500)],
    *,
    max_results: Annotated[int, Query(ge=1, le=20)] = 10,
    include_answer: bool = True,
) -> APIResponse[SearchResponse]:
    """Search remains an immediate Tavily operation for LLM/tool use."""
    owner = get_client_identifier(request)
    is_allowed, rate_info = await rate_limiter.acquire(owner, RateLimitScope.SEARCH)
    if not is_allowed:
        raise TooManyRequestsException(
            detail=rate_info.get("error") or "Rate limit exceeded",
            data={"retry_after": rate_info.get("retry_after")},
        )
    result = await service.search(
        SearchRequest(query=query, max_results=max_results, include_answer=include_answer)
    )
    return render_result(result, response, message="Search completed")


@router.get(path="/rate-limit")
async def get_rate_limit_info(
    request: Request,
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> RateLimitInfo:
    owner = get_client_identifier(request)
    crawl_remaining = await rate_limiter.get_remaining(owner, RateLimitScope.CRAWL)
    search_remaining = await rate_limiter.get_remaining(owner, RateLimitScope.SEARCH)
    return RateLimitInfo(
        remaining_minute=min(crawl_remaining["remaining_minute"], search_remaining["remaining_minute"]),
        remaining_hour=min(crawl_remaining["remaining_hour"], search_remaining["remaining_hour"]),
    )
