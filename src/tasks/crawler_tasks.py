"""Celery execution boundary for durable Crawl4AI jobs."""

from __future__ import annotations

import asyncio

from returns.result import Failure

from app.config import get_settings
from app.connections.celery import CeleryTaskPayload, CeleryTaskRegistry, ResilientTask, celery_app
from app.connections.celery_task_names import CRAWLER_CRAWL
from app.connections.crawl4ai import close_crawl4ai_crawler, create_crawl4ai_crawler
from app.connections.redis import create_redis_client
from app.features.crawler.dto import CrawlRequest
from app.features.crawler.job_store import CrawlJobStore
from app.features.crawler.service import CrawlerService
from app.shared.crawler import WebCrawler
from app.shared.crawler.processor import get_processor
from app.utils import logger


class CrawlerJobPayload(CeleryTaskPayload):
    """Validated wire payload for a durable crawler task."""

    crawl_id: str
    request_payload: dict[str, object]


CeleryTaskRegistry.register(CRAWLER_CRAWL, CrawlerJobPayload)


async def _run_crawl_job(crawl_id: str, request_payload: dict[str, object]) -> None:
    settings = get_settings()
    redis = create_redis_client(settings.REDIS_URL)
    store = CrawlJobStore(redis)
    browser = None
    try:
        if await store.is_cancelled(crawl_id):
            await store.mark_cancelled(crawl_id)
            return
        await store.mark_running(crawl_id)
        request = CrawlRequest.model_validate(request_payload)
        browser = await create_crawl4ai_crawler()
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
        logger.bind(crawl_id=crawl_id).exception("Durable crawler job failed")
        try:
            await store.mark_failed(crawl_id, str(exc))
        except (ConnectionError, OSError, RuntimeError, ValueError) as store_exc:
            store_exc.add_note(f"crawl_id={crawl_id}, operation=mark_failed")
            logger.bind(crawl_id=crawl_id).exception("Could not persist crawler failure")
        raise
    finally:
        if browser is not None:
            await close_crawl4ai_crawler(browser)
        await redis.aclose(close_connection_pool=True)


@celery_app.task(name=CRAWLER_CRAWL, bind=True, base=ResilientTask)
def crawl_job(
    self: ResilientTask,
    *,
    crawl_id: str,
    request_payload: dict[str, object],
) -> dict[str, str]:
    """Execute one durable crawl in a worker-owned browser lifecycle."""
    _ = self
    asyncio.run(_run_crawl_job(crawl_id, request_payload))
    return {"crawl_id": crawl_id, "status": "finished"}
