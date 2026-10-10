"""Celery execution boundary for durable Crawl4AI jobs."""

from __future__ import annotations

from returns.result import Failure

from app.config import get_settings
from app.connections.redis import create_redis_client
from app.features.crawler.crawler import WebCrawler
from app.features.crawler.dto import CrawlRequest
from app.features.crawler.job_store import CrawlJobStore
from app.features.crawler.processor import get_processor
from app.features.crawler.service import CrawlerService
from app.observability.logging import logger
from app.shared.jobs.names import CRAWLER_CRAWL
from app.worker.app import celery_app
from app.worker.crawler import get_crawler_worker_resources, run_on_crawler_worker_loop
from app.worker.task import ResilientTask


async def _run_crawl_job(crawl_id: str, request_payload: dict[str, object]) -> None:
    settings = get_settings()
    redis = create_redis_client(settings.REDIS_URL)
    store = CrawlJobStore(redis)
    try:
        if await store.is_cancelled(crawl_id):
            await store.mark_cancelled(crawl_id)
            return
        await store.mark_running(crawl_id)
        request = CrawlRequest.model_validate(request_payload)
        browser = get_crawler_worker_resources().browser
        processor = await get_processor()
        crawler = WebCrawler(redis_client=redis, browser=browser)
        service = CrawlerService(crawler=crawler, processor=processor, redis_client=redis)
        result = await service.crawl(request, crawl_id=crawl_id)
        if await store.is_cancelled(crawl_id):
            await store.mark_cancelled(crawl_id)
            return
        if isinstance(result, Failure):
            await store.mark_failed(crawl_id, result.failure().message)
            return
        await store.save_result(crawl_id, result.unwrap())
    except Exception as exc:
        exc.add_note(f"crawl_id={crawl_id}, operation=crawl_job")
        logger.bind(crawl_id=crawl_id).exception("Durable crawler job failed")
        try:
            await store.mark_failed(crawl_id, str(exc))
        except (ConnectionError, OSError, RuntimeError, ValueError) as store_exc:
            store_exc.add_note(f"crawl_id={crawl_id}, operation=mark_failed")
            logger.bind(crawl_id=crawl_id).exception("Could not persist crawler failure")
        raise
    finally:
        # The browser is owned by the worker process, not the task: per-task
        # cleanup closes only the per-task Redis client.
        await redis.aclose(close_connection_pool=True)


@celery_app.task(name=CRAWLER_CRAWL, base=ResilientTask)
def crawl_job(
    *,
    crawl_id: str,
    request_payload: dict[str, object],
) -> dict[str, str]:
    """Execute one durable crawl in a worker-owned browser lifecycle."""
    run_on_crawler_worker_loop(lambda: _run_crawl_job(crawl_id, request_payload))
    return {"crawl_id": crawl_id, "status": "finished"}
