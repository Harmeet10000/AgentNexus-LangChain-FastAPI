"""Celery execution boundary for durable Crawl4AI jobs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import NoReturn

from playwright.async_api import Error as PlaywrightError
from redis.exceptions import RedisError
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
from app.utils import InfrastructureException, ServiceUnavailableException
from app.worker.celery.app import get_worker_celery_app
from app.worker.crawler import (
    aprobe_crawler_browser,
    areset_crawler_worker,
    get_crawler_worker_resources,
    probe_crawler_browser,
    reset_crawler_worker,
    run_on_crawler_worker_loop,
)
from app.worker.task import ResilientTask

celery_app = get_worker_celery_app()

_RETRYABLE_EXC: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    ServiceUnavailableException,
    InfrastructureException,
    RedisError,
)

_STORE_WRITE_EXC: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    RuntimeError,
    ValueError,
    RedisError,
)


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE_EXC)


def _unhealthy_browser_failure(failure_message: str, crawl_id: str) -> NoReturn:
    """Raise a retryable failure for a browser-level crawl failure."""
    raise InfrastructureException(
        detail=f"Crawler browser unhealthy: {failure_message}",
        retryable=True,
        data={"crawl_id": crawl_id},
    )


async def _mark_terminal(store: CrawlJobStore, crawl_id: str, message: str) -> None:
    """Write FAILED unless the job was cancelled, which wins over failure."""
    try:
        if await store.is_cancelled(crawl_id):
            await store.mark_cancelled(crawl_id)
            return
    except _STORE_WRITE_EXC as cancel_exc:
        cancel_exc.add_note(f"crawl_id={crawl_id}, operation=mark_terminal_cancel_check")
        logger.bind(crawl_id=crawl_id).warning("crawler_cancel_check_failed")
    try:
        await store.mark_failed(crawl_id, message)
    except _STORE_WRITE_EXC as store_exc:
        store_exc.add_note(f"crawl_id={crawl_id}, operation=mark_failed")
        logger.bind(crawl_id=crawl_id).exception("Could not persist crawler failure")


async def _mark_failed_best_effort(crawl_id: str, message: str) -> None:
    """Persist a terminal job status without the worker-owned browser loop."""
    settings = get_settings()
    redis = create_redis_client(settings.REDIS_URL)
    try:
        await _mark_terminal(CrawlJobStore(redis), crawl_id, message)
    except Exception as store_exc:  # noqa: BLE001 — best-effort terminal write
        store_exc.add_note(f"crawl_id={crawl_id}, operation=mark_failed_best_effort")
        logger.bind(crawl_id=crawl_id).warning("Could not persist crawler failure")
    finally:
        await redis.aclose(close_connection_pool=True)


def _mark_failed_sync(crawl_id: str, message: str) -> None:
    try:
        asyncio.run(_mark_failed_best_effort(crawl_id, message))
    except Exception as exc:  # noqa: BLE001 — terminal write must never mask retry
        exc.add_note(f"crawl_id={crawl_id}, operation=mark_failed_sync")
        logger.bind(crawl_id=crawl_id).warning("crawler_failure_write_failed")


async def _run_crawl_job(
    crawl_id: str, request_payload: dict[str, object], *, final_attempt: bool
) -> None:
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
            failure_message = result.failure().message
            if not await aprobe_crawler_browser():
                # Browser-level failure disguised as a result: the crawler
                # converts Playwright errors into Failure, so a dead browser
                # would otherwise fail every later job without ever tripping
                # the task-boundary probe. Reset on the owning loop and raise
                # retryable so the retry re-provisions a fresh browser.
                await areset_crawler_worker()
                _unhealthy_browser_failure(failure_message, crawl_id)
            await _mark_terminal(store, crawl_id, failure_message)
            return
        await store.save_result(crawl_id, result.unwrap())
    except Exception as exc:
        exc.add_note(f"crawl_id={crawl_id}, operation=crawl_job")
        logger.bind(crawl_id=crawl_id).exception("Durable crawler job failed")
        if _is_retryable(exc) and not final_attempt:
            raise
        await _mark_terminal(store, crawl_id, str(exc))
        raise
    finally:
        # The browser is owned by the worker process, not the task: per-task
        # cleanup closes only the per-task Redis client.
        await redis.aclose(close_connection_pool=True)


@celery_app.task(
    name=CRAWLER_CRAWL,
    bind=True,
    base=ResilientTask,
    autoretry_for=(
        *ResilientTask.autoretry_for,
        ServiceUnavailableException,
        InfrastructureException,
        RedisError,
    ),
    soft_time_limit=330,
    time_limit=360,
)
def crawl_job(
    self: ResilientTask,
    *,
    crawl_id: str,
    request_payload: dict[str, object],
) -> dict[str, str]:
    """Execute one durable crawl in a worker-owned browser lifecycle."""
    settings = get_settings()
    final_attempt = self.request.retries >= settings.CELERY_RETRY_MAX_RETRIES
    try:
        run_on_crawler_worker_loop(
            lambda: _run_crawl_job(crawl_id, request_payload, final_attempt=final_attempt)
        )
    except PlaywrightError:
        if not probe_crawler_browser():
            reset_crawler_worker()
        raise
    except ServiceUnavailableException as exc:
        # Provisioning failed before the job coroutine started, so the job was
        # never marked running/failed. Resolve it on the terminal attempt.
        if final_attempt:
            _mark_failed_sync(crawl_id, str(exc))
        raise
    return {"crawl_id": crawl_id, "status": "finished"}
