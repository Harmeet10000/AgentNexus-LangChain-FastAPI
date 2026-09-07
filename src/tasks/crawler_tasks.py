"""Celery execution boundary for durable Crawl4AI jobs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from celery.signals import worker_process_init, worker_process_shutdown
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

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any

    from crawl4ai import AsyncWebCrawler


class CrawlerJobPayload(CeleryTaskPayload):
    """Validated wire payload for a durable crawler task."""

    crawl_id: str
    request_payload: dict[str, object]


CeleryTaskRegistry.register(CRAWLER_CRAWL, CrawlerJobPayload)


_WORKER_LOOP: asyncio.AbstractEventLoop | None = None
_WORKER_BROWSER: AsyncWebCrawler | None = None


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent event loop for this worker process.

    Playwright browsers are bound to the loop that created them, so every
    task in the same prefork process must share one loop. ``asyncio.run()``
    per task would create a fresh loop and prevent browser reuse.
    """
    global _WORKER_LOOP  # noqa: PLW0603
    if _WORKER_LOOP is None or _WORKER_LOOP.is_closed():
        _WORKER_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_WORKER_LOOP)
    return _WORKER_LOOP


async def _get_shared_browser() -> AsyncWebCrawler:
    """Return the per-worker browser, creating it lazily on first use."""
    global _WORKER_BROWSER  # noqa: PLW0603
    if _WORKER_BROWSER is None:
        _WORKER_BROWSER = await create_crawl4ai_crawler()
    return _WORKER_BROWSER


def _run_on_worker_loop(
    coro: Coroutine[Any, Any, None],
) -> None:
    _get_worker_loop().run_until_complete(coro)


@worker_process_init.connect
def _reset_crawler_worker_state(**_kwargs: object) -> None:
    """Drop fork-inherited loop/browser handles in each child process."""
    global _WORKER_LOOP, _WORKER_BROWSER  # noqa: PLW0603
    _WORKER_LOOP = None
    _WORKER_BROWSER = None


@worker_process_shutdown.connect
def _close_crawler_worker_state(**_kwargs: object) -> None:
    """Close the per-worker browser and loop during worker shutdown."""
    global _WORKER_LOOP, _WORKER_BROWSER  # noqa: PLW0603
    try:
        if (
            _WORKER_BROWSER is not None
            and _WORKER_LOOP is not None
            and not _WORKER_LOOP.is_closed()
        ):
            _WORKER_LOOP.run_until_complete(close_crawl4ai_crawler(_WORKER_BROWSER))
    except (OSError, RuntimeError):
        logger.exception("Could not close per-worker crawler browser")
    finally:
        _WORKER_BROWSER = None
        if _WORKER_LOOP is not None and not _WORKER_LOOP.is_closed():
            _WORKER_LOOP.close()
        _WORKER_LOOP = None


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
        browser = await _get_shared_browser()
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
        # The browser is owned by the worker process, not the task: per-task
        # cleanup closes only the per-task Redis client.
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
    _run_on_worker_loop(_run_crawl_job(crawl_id, request_payload))
    return {"crawl_id": crawl_id, "status": "finished"}
