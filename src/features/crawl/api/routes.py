"""Web crawling API routes."""

from typing import List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from pydantic import HttpUrl

from features.crawl.api.schemas import (
    CrawlRequest,
    SmartCrawlRequest,
    CrawlStatus,
    CrawlResult,
    SmartCrawlResponse,
    UrlExtractionRequest,
    UrlExtractionResponse,
    SitemapResponse,
    CrawlJobCancelResponse,
)
from features.crawl.services.crawl_service import CrawlService, get_crawl_service
from core.config.logging_config import LoggerAdapter
from shared.schemas.response import (
    http_success,
    http_error,
)

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/crawl", tags=["Web Crawling"])


@router.post("/", response_model=CrawlStatus)
async def crawl_website(
    request: Request,
    background_tasks: BackgroundTasks,
    crawl_request: CrawlRequest,
    service: CrawlService = Depends(get_crawl_service),
):
    """Crawl a website and extract content."""
    try:
        result = await service.crawl_website(crawl_request, background_tasks)

        return http_success(
            request, message="Crawl job started successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to start crawl job", error=str(e))
        return http_error(request, e, 500)


@router.post("/smart", response_model=SmartCrawlResponse)
async def smart_crawl(
    request: Request,
    smart_crawl_request: SmartCrawlRequest,
    service: CrawlService = Depends(get_crawl_service),
):
    """Perform intelligent web crawling with AI extraction."""
    try:
        result = await service.smart_crawl(smart_crawl_request)

        return http_success(
            request, message="Smart crawl completed successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to perform smart crawl", error=str(e))
        return http_error(request, e, 500)


@router.get("/status/{job_id}", response_model=CrawlStatus)
async def get_crawl_status(
    request: Request, job_id: str, service: CrawlService = Depends(get_crawl_service)
):
    """Get crawl job status."""
    try:
        result = await service.get_crawl_status(job_id)

        if not result:
            return http_error(
                request, Exception("Crawl job not found"), status_code=404
            )

        return http_success(
            request, message="Crawl status retrieved successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to get crawl status", error=str(e))
        return http_error(request, e, 500)


@router.get("/results/{job_id}", response_model=List[CrawlResult])
async def get_crawl_results(
    request: Request,
    job_id: str,
    limit: int = 100,
    service: CrawlService = Depends(get_crawl_service),
):
    """Get crawl job results."""
    try:
        result = await service.get_crawl_results(job_id, limit)

        return http_success(
            request, message="Crawl results retrieved successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to get crawl results", error=str(e))
        return http_error(request, e, 500)


@router.post("/extract-from-url", response_model=UrlExtractionResponse)
async def extract_from_url(
    request: Request,
    extraction_request: UrlExtractionRequest,
    service: CrawlService = Depends(get_crawl_service),
):
    """Extract specific content from a URL."""
    try:
        result = await service.extract_from_url(extraction_request)

        return http_success(
            request, message="Content extracted successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to extract from URL", error=str(e))
        return http_error(request, e, 500)


@router.post("/sitemap", response_model=SitemapResponse)
async def parse_sitemap(
    request: Request, url: HttpUrl, service: CrawlService = Depends(get_crawl_service)
):
    """Parse and extract URLs from a sitemap."""
    try:
        result = await service.parse_sitemap(url)

        return http_success(request, message="Sitemap parsed successfully", data=result)

    except Exception as e:
        logger.error("Failed to parse sitemap", error=str(e))
        return http_error(request, e, 500)


@router.delete("/job/{job_id}", response_model=CrawlJobCancelResponse)
async def cancel_crawl_job(
    request: Request, job_id: str, service: CrawlService = Depends(get_crawl_service)
):
    """Cancel a running crawl job."""
    try:
        result = await service.cancel_crawl_job(job_id)

        return http_success(
            request, message=f"Crawl job {job_id} cancelled successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to cancel crawl job", error=str(e))
        return http_error(request, e, 500)
