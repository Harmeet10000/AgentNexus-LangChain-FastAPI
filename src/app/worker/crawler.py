"""Per-child crawler resources, provisioned on first task use."""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import TYPE_CHECKING

from celery.signals import worker_process_init, worker_process_shutdown, worker_shutdown
from crawl4ai import AsyncWebCrawler
from pydantic import BaseModel, ConfigDict

from app.connections.crawl4ai import close_crawl4ai_crawler, create_crawl4ai_crawler
from app.observability.logging import logger
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any


class CrawlerWorkerResources(BaseModel):
    """Browser bound to one worker child and its persistent event loop."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    browser: AsyncWebCrawler


_WORKER_PID: int | None = None
_WORKER_RUNNER: asyncio.Runner | None = None
_WORKER_RESOURCES: CrawlerWorkerResources | None = None


async def _provision_crawler_worker() -> CrawlerWorkerResources:
    """Create the per-worker browser on the loop that will execute tasks."""
    return CrawlerWorkerResources(browser=await create_crawl4ai_crawler())


@worker_process_init.connect
def initialize_crawler_worker(**_kwargs: object) -> None:
    """Reset inherited state; first task provisions resources after child startup."""
    global _WORKER_RESOURCES, _WORKER_RUNNER, _WORKER_PID  # noqa: PLW0603
    _WORKER_RESOURCES = None
    _WORKER_RUNNER = None
    _WORKER_PID = os.getpid()


def _ensure_crawler_worker() -> None:
    """Lazily provision once on the owning child's persistent loop."""
    global _WORKER_RESOURCES, _WORKER_RUNNER  # noqa: PLW0603
    if os.getpid() != _WORKER_PID:
        initialize_crawler_worker()
    if _WORKER_RESOURCES is not None:
        return
    if _WORKER_RUNNER is None:
        _WORKER_RUNNER = asyncio.Runner()
    try:
        _WORKER_RESOURCES = _WORKER_RUNNER.run(_provision_crawler_worker())
    except Exception as exc:
        exc.add_note("capability=crawler, operation=worker_provision")
        _WORKER_RUNNER.close()
        _WORKER_RUNNER = None
        raise ServiceUnavailableException(detail="Crawler worker provisioning failed") from exc
    else:
        logger.info("Crawler worker resources initialized")


def get_crawler_worker_resources() -> CrawlerWorkerResources:
    """Return provisioned resources or a typed, capability-naming failure."""
    _ensure_crawler_worker()
    if _WORKER_RESOURCES is None:
        raise ServiceUnavailableException(
            detail="Crawler worker is unavailable",
            data={"capability": "crawler_browser"},
        )
    return _WORKER_RESOURCES


def run_on_crawler_worker_loop[T](
    coroutine_factory: Callable[[], Coroutine[Any, Any, T]],
) -> T:
    """Run task I/O on the same loop that created the process resources.

    A soft time limit escapes ``run_until_complete`` without cancelling the
    coroutine, which would otherwise resume on this loop alongside the next
    task. Cancel and drain it before the next task runs.
    """
    get_crawler_worker_resources()
    if _WORKER_RUNNER is None:
        message = "Worker resource guard returned without a runner"
        raise AssertionError(message)
    loop = _WORKER_RUNNER.get_loop()
    task = loop.create_task(coroutine_factory())
    try:
        return loop.run_until_complete(task)
    except BaseException:
        task.cancel()
        with contextlib.suppress(BaseException):
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        raise


async def _probe_crawler_browser() -> bool:
    """Return True when the shared browser still executes a trivial load."""
    resources = _WORKER_RESOURCES
    if resources is None:
        return False
    try:
        from typing import cast  # noqa: PLC0415 — probe-only cast

        from crawl4ai import CacheMode, CrawlerRunConfig  # noqa: PLC0415 — probe-only import

        async with asyncio.timeout(10):
            await cast("Any", resources.browser).arun(
                url="about:blank",
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
            )
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        exc.add_note("operation=crawler_browser_probe")
        logger.bind(error_type=type(exc).__name__).warning("crawler_browser_probe_failed")
        return False
    return True


def probe_crawler_browser() -> bool:
    """Synchronously probe the shared browser on its owning loop."""
    runner = _WORKER_RUNNER
    if runner is None or _WORKER_RESOURCES is None:
        return False
    try:
        return runner.run(_probe_crawler_browser())
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        exc.add_note("operation=crawler_browser_probe")
        return False


async def aprobe_crawler_browser() -> bool:
    """Probe the shared browser from inside its owning loop."""
    if _WORKER_RESOURCES is None:
        return False
    return await _probe_crawler_browser()


async def areset_crawler_worker() -> None:
    """Discard the shared browser from inside its owning loop.

    Unlike :func:`reset_crawler_worker`, this must not touch the runner:
    the loop is running, so it can neither run a close coroutine through
    ``runner.run`` nor be closed. The next ``_ensure`` re-provisions the
    browser on the same runner.
    """
    global _WORKER_RESOURCES  # noqa: PLW0603
    resources = _WORKER_RESOURCES
    _WORKER_RESOURCES = None
    if resources is not None:
        await _close_crawler_browser(resources)


def reset_crawler_worker() -> None:
    """Discard the shared browser and runner; the next task re-provisions.

    Sync contexts only (task boundary, shutdown): the loop must not be
    running. From inside the owning loop, use :func:`areset_crawler_worker`.
    """
    global _WORKER_RESOURCES, _WORKER_RUNNER  # noqa: PLW0603
    resources = _WORKER_RESOURCES
    runner = _WORKER_RUNNER
    _WORKER_RESOURCES = None
    _WORKER_RUNNER = None
    if runner is not None and resources is not None:
        try:
            runner.run(_close_crawler_browser(resources))
        except Exception as exc:  # noqa: BLE001 — reset must never raise
            exc.add_note("operation=reset_crawler_worker")
            logger.bind(error_type=type(exc).__name__).warning("crawler_browser_reset_close_failed")
    if runner is not None:
        runner.close()


async def _close_crawler_browser(resources: CrawlerWorkerResources) -> None:
    """Close the browser with a deadline so shutdown and reset never hang."""
    try:
        async with asyncio.timeout(10):
            await close_crawl4ai_crawler(resources.browser)
    except Exception as exc:  # noqa: BLE001 — shutdown must never raise
        exc.add_note("operation=shutdown_crawler_worker")
        logger.bind(error_type=type(exc).__name__).warning("crawler_browser_close_failed")


@worker_process_shutdown.connect
@worker_shutdown.connect
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
        runner.run(_close_crawler_browser(resources))
        logger.info("Crawler worker resources closed")
    except Exception as exc:  # noqa: BLE001 — shutdown must never raise
        exc.add_note("operation=shutdown_crawler_worker")
        logger.bind(error_type=type(exc).__name__).warning("crawler_browser_close_failed")
    finally:
        if runner is not None:
            runner.close()
