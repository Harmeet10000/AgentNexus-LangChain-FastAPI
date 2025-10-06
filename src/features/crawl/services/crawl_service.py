"""Crawl service implementation."""

from typing import List, Dict, Any
from datetime import datetime
from fastapi import Depends, HTTPException, BackgroundTasks
from pydantic import HttpUrl
from shared.utils.id_generator import generate_job_id

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
from features.crawl.repositories.crawl_repository import (
    CrawlRepository,
    get_crawl_repository,
)
from features.rag.repositories.vector_repository import (
    VectorRepository,
    get_vector_repository,
)
from core.config.logging_config import LoggerAdapter
from core.cache.redis_client import redis_cache
from services.crawl4ai.crawler import web_crawler

logger = LoggerAdapter(__name__)


class CrawlService:
    """Service for web crawling operations."""

    def __init__(self, crawl_repo: CrawlRepository, vector_repo: VectorRepository):
        """Initialize crawl service."""
        self.crawl_repo = crawl_repo
        self.vector_repo = vector_repo

    async def crawl_website(
        self, crawl_request: CrawlRequest, background_tasks: BackgroundTasks
    ) -> CrawlStatus:
        """Crawl a website and extract content."""
        try:
            # Generate job ID using nanoid
            job_id = generate_job_id("crawl")

            # Create job status
            job_status = {
                "job_id": job_id,
                "status": "pending",
                "url": str(crawl_request.url),
                "pages_crawled": 0,
                "start_time": datetime.now().isoformat(),
            }

            # Store job status in cache
            await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)  # 1 hour

            # Save to repository
            await self.crawl_repo.save_crawl_job(
                {
                    "job_id": job_id,
                    "url": str(crawl_request.url),
                    "status": "pending",
                    "config": crawl_request.dict(),
                }
            )

            # Start crawling in background
            background_tasks.add_task(
                self._crawl_website_background, job_id, crawl_request
            )

            job_status["status"] = "running"
            await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)

            logger.info(f"Crawl job started: {job_id}")

            return CrawlStatus(**job_status)

        except Exception as e:
            logger.error("Failed to start crawl job", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def _crawl_website_background(self, job_id: str, request: CrawlRequest):
        """Background task to crawl website."""
        try:
            # Update job status
            await self.crawl_repo.update_crawl_status(job_id, "running")

            job_status = await redis_cache.get(f"crawl_job:{job_id}")
            if job_status:
                job_status["status"] = "running"
                await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)

            # Perform crawling
            results = await web_crawler.crawl(
                url=str(request.url),
                max_depth=request.max_depth,
                max_pages=request.max_pages,
                extract_content=request.extract_content,
                extract_links=request.extract_links,
                extract_images=request.extract_images,
                follow_robots_txt=request.follow_robots_txt,
                wait_time=request.wait_time,
                user_agent=request.user_agent,
            )

            # Save results to repository
            await self.crawl_repo.save_crawl_results(job_id, results)

            # Save to vector store if requested
            if request.save_to_vectors and results:
                documents = []
                for result in results:
                    if result.get("content"):
                        doc = {
                            "content": result["content"],
                            "metadata": {
                                "url": result["url"],
                                "title": result.get("title", ""),
                                "crawled_at": result.get("timestamp", ""),
                                "source": "web_crawl",
                                "job_id": job_id,
                            },
                        }
                        documents.append(doc)

                if documents:
                    await self.vector_repo.add_documents(
                        documents, namespace=request.namespace
                    )

            # Update job status
            results_summary = {
                "total_pages": len(results),
                "total_links": sum(len(r.get("links", [])) for r in results),
                "saved_to_vectors": request.save_to_vectors,
            }

            await self.crawl_repo.update_crawl_status(
                job_id, "completed", results_summary=results_summary
            )

            if job_status:
                job_status["status"] = "completed"
                job_status["pages_crawled"] = len(results)
                job_status["end_time"] = datetime.now().isoformat()
                job_status["results_summary"] = results_summary
                await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)

            # Store results in cache
            await redis_cache.set(f"crawl_results:{job_id}", results, ttl=3600)

            logger.info(f"Crawl job completed: {job_id}, {len(results)} pages")

        except Exception as e:
            logger.error(f"Failed to complete crawl job {job_id}", error=str(e))

            # Update job status with error
            await self.crawl_repo.update_crawl_status(
                job_id, "failed", error_message=str(e)
            )

            job_status = await redis_cache.get(f"crawl_job:{job_id}")
            if job_status:
                job_status["status"] = "failed"
                job_status["error"] = str(e)
                job_status["end_time"] = datetime.now().isoformat()
                await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)

    async def smart_crawl(self, request: SmartCrawlRequest) -> SmartCrawlResponse:
        """Perform intelligent web crawling with AI extraction."""
        try:
            # Use smart scraper for intelligent extraction
            result = await smart_scraper.scrape_with_ai(
                url=str(request.url),
                extraction_prompt=request.extraction_prompt,
                css_selector=request.css_selector,
                wait_for_selector=request.wait_for_selector,
                javascript_enabled=request.javascript_enabled,
                screenshot=request.screenshot,
                extract_structured_data=request.extract_structured_data,
            )

            logger.info(f"Smart crawl completed for: {request.url}")

            return SmartCrawlResponse(
                url=str(request.url),
                extracted_data=result.get("extracted_data"),
                structured_data=result.get("structured_data"),
                screenshot=result.get("screenshot"),
                metadata=result.get("metadata"),
            )

        except Exception as e:
            logger.error("Failed to perform smart crawl", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def get_crawl_status(self, job_id: str) -> CrawlStatus:
        """Get crawl job status."""
        try:
            job_status = await redis_cache.get(f"crawl_job:{job_id}")

            if not job_status:
                # Try to get from repository
                job_data = await self.crawl_repo.get_crawl_job(job_id)
                if not job_data:
                    return None

                job_status = {
                    "job_id": job_data["job_id"],
                    "status": job_data["status"],
                    "url": job_data["url"],
                    "pages_crawled": job_data.get("pages_crawled", 0),
                    "start_time": job_data["created_at"].isoformat(),
                    "end_time": (
                        job_data.get("updated_at", {}).isoformat()
                        if job_data.get("updated_at")
                        else None
                    ),
                    "error": job_data.get("error_message"),
                    "results_summary": job_data.get("results_summary"),
                }

            return CrawlStatus(**job_status)

        except Exception as e:
            logger.error("Failed to get crawl status", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def get_crawl_results(
        self, job_id: str, limit: int = 100
    ) -> List[CrawlResult]:
        """Get crawl job results."""
        try:
            # Check if job exists
            job_status = await self.get_crawl_status(job_id)
            if not job_status:
                raise HTTPException(status_code=404, detail="Crawl job not found")

            # Get results from cache first
            results = await redis_cache.get(f"crawl_results:{job_id}")

            if not results:
                # Get from repository
                results = await self.crawl_repo.get_crawl_results(job_id, limit)

                if not results:
                    if job_status.status == "running":
                        return []  # Job still running
                    raise HTTPException(status_code=404, detail="No results found")

            # Convert to response model
            crawl_results = []
            for result in results[:limit]:
                crawl_results.append(CrawlResult(**result))

            return crawl_results

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get crawl results", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def extract_from_url(
        self, request: UrlExtractionRequest
    ) -> UrlExtractionResponse:
        """Extract specific content from a URL."""
        try:
            result = await web_crawler.extract_specific_content(
                url=str(request.url),
                css_selectors=request.css_selectors,
                xpath_queries=request.xpath_queries,
            )

            return UrlExtractionResponse(
                url=str(request.url),
                extracted_content=result,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error("Failed to extract from URL", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def parse_sitemap(self, url: HttpUrl) -> SitemapResponse:
        """Parse and extract URLs from a sitemap."""
        try:
            urls = await web_crawler.parse_sitemap(str(url))

            return SitemapResponse(
                sitemap_url=str(url), total_urls=len(urls), urls=urls
            )

        except Exception as e:
            logger.error("Failed to parse sitemap", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    async def cancel_crawl_job(self, job_id: str) -> CrawlJobCancelResponse:
        """Cancel a running crawl job."""
        try:
            job_status = await redis_cache.get(f"crawl_job:{job_id}")

            if not job_status:
                raise HTTPException(status_code=404, detail="Crawl job not found")

            if job_status["status"] in ["completed", "failed"]:
                raise HTTPException(
                    status_code=400, detail=f"Job already {job_status['status']}"
                )

            # Update job status
            job_status["status"] = "cancelled"
            job_status["end_time"] = datetime.now().isoformat()
            await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)

            # Update repository
            await self.crawl_repo.update_crawl_status(job_id, "cancelled")

            # Clean up results if any
            await redis_cache.delete(f"crawl_results:{job_id}")

            logger.info(f"Crawl job cancelled: {job_id}")

            return CrawlJobCancelResponse(
                message=f"Crawl job {job_id} cancelled successfully", job_id=job_id
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to cancel crawl job", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


def get_crawl_service(
    crawl_repo: CrawlRepository = Depends(get_crawl_repository),
    vector_repo: VectorRepository = Depends(get_vector_repository),
) -> CrawlService:
    """Dependency to get crawl service."""
    return CrawlService(crawl_repo, vector_repo)
