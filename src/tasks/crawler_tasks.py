"""Celery execution boundary for durable Crawl4AI jobs."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from celery.signals import worker_process_init, worker_process_shutdown
from crawl4ai import AsyncWebCrawler  # noqa: TC002 — resolved at runtime by Pydantic
from pydantic import BaseModel, ConfigDict
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
from app.utils import ServiceUnavailableException, logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence
    from typing import Any


class CrawlerJobPayload(CeleryTaskPayload):
    """Validated wire payload for a durable crawler task."""

    crawl_id: str
    request_payload: dict[str, object]


CeleryTaskRegistry.register(CRAWLER_CRAWL, CrawlerJobPayload)


class CrawlerWorkerResources(BaseModel):
    """Browser bound to one worker child and its persistent event loop."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    browser: AsyncWebCrawler


_WORKER_RUNNER: asyncio.Runner | None = None
_WORKER_RESOURCES: CrawlerWorkerResources | None = None


def _worker_consumes_queue(queue_name: str, argv: Sequence[str] | None = None) -> bool:
    """Return whether this worker command consumes ``queue_name``.

    Celery does not pass the consumer object to ``worker_process_init``. Pool
    children do retain the worker command line, so an explicit ``-Q``/``--queues``
    is the narrow source of truth. A command without a queue flag consumes the
    app's configured queues and must therefore provision conservatively.
    """
    arguments = list(argv if argv is not None else sys.argv)
    configured: list[str] = []
    for index, argument in enumerate(arguments):
        if argument in {"-Q", "--queues"} and index + 1 < len(arguments):
            configured.extend(arguments[index + 1].split(","))
        elif argument.startswith("--queues="):
            configured.extend(argument.partition("=")[2].split(","))
        elif argument.startswith("-Q") and len(argument) > 2:
            configured.extend(argument[2:].split(","))
    if not configured:
        return True
    return queue_name in {name.strip() for name in configured if name.strip()}


async def _provision_crawler_worker() -> CrawlerWorkerResources:
    """Create the per-worker browser on the loop that will execute tasks."""
    return CrawlerWorkerResources(browser=await create_crawl4ai_crawler())


@worker_process_init.connect
def initialize_crawler_worker(**_kwargs: object) -> None:
    """Provision once in each forked child, never in the Celery parent."""
    global _WORKER_RESOURCES, _WORKER_RUNNER  # noqa: PLW0603
    _WORKER_RESOURCES = None
    crawler_queue = getattr(get_settings(), "CELERY_CRAWLER_QUEUE", "crawler")
    if not _worker_consumes_queue(crawler_queue):
        logger.bind(queue=crawler_queue).info(
            "Crawler worker resources skipped for unrelated worker"
        )
        return
    _WORKER_RUNNER = asyncio.Runner()
    try:
        _WORKER_RESOURCES = _WORKER_RUNNER.run(_provision_crawler_worker())
    except Exception as exc:  # noqa: BLE001 — failed optional capability degrades the worker
        exc.add_note("capability=crawler_browser, operation=provision")
        logger.bind(error_type=type(exc).__name__).exception("Crawler worker provisioning failed")
        _WORKER_RUNNER.close()
        _WORKER_RUNNER = None
    else:
        logger.info("Crawler worker resources initialized")


def get_crawler_worker_resources() -> CrawlerWorkerResources:
    """Return provisioned resources or a typed, capability-naming failure."""
    if _WORKER_RESOURCES is None:
        raise ServiceUnavailableException(
            detail="Crawler worker is unavailable",
            data={"capability": "crawler_browser"},
        )
    return _WORKER_RESOURCES


def run_on_crawler_worker_loop[T](
    coroutine_factory: Callable[[], Coroutine[Any, Any, T]],
) -> T:
    """Run task I/O on the same loop that created the process resources."""
    if _WORKER_RUNNER is None:
        get_crawler_worker_resources()
        message = "Worker resource guard returned without a runner"
        raise AssertionError(message)
    return _WORKER_RUNNER.run(coroutine_factory())


@worker_process_shutdown.connect
def shutdown_crawler_worker(**_kwargs: object) -> None:
    """Release the browser once and close the persistent loop."""
    global _WORKER_RESOURCES, _WORKER_RUNNER  # noqa: PLW0603
    resources = _WORKER_RESOURCES
    runner = _WORKER_RUNNER
    _WORKER_RESOURCES = None
    _WORKER_RUNNER = None
    if resources is None:
        logger.info("Crawler worker had no resources to close")
        if runner is not None:
            runner.close()
        return
    try:
        if runner is None:
            logger.error("Crawler worker loop absent during resource shutdown")
            return
        runner.run(close_crawl4ai_crawler(resources.browser))
        logger.info("Crawler worker resources closed")
    except (OSError, RuntimeError) as exc:
        exc.add_note("operation=shutdown_crawler_worker")
        logger.exception("Could not close per-worker crawler browser")
    finally:
        if runner is not None:
            runner.close()


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
