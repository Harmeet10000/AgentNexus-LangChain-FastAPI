"""Per-child crawler resources, provisioned on first task use."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, override

from celery.signals import worker_process_init, worker_process_shutdown, worker_shutdown
from crawl4ai import AsyncWebCrawler
from pydantic import BaseModel, ConfigDict

from app.connections.crawl4ai import close_crawl4ai_crawler, create_crawl4ai_crawler
from app.observability.logging import logger
from app.utils import ServiceUnavailableException
from app.worker.capability_runtime import (
    CapabilityLifecycle,
    CapabilityOwner,
    CapabilityUsageError,
)

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


async def _close_crawler_browser(resources: CrawlerWorkerResources) -> None:
    """Close the browser with a deadline so shutdown and reset never hang."""
    try:
        async with asyncio.timeout(10):
            await close_crawl4ai_crawler(resources.browser)
    except Exception as exc:  # noqa: BLE001 — shutdown must never raise
        exc.add_note("operation=shutdown_crawler_worker")
        logger.bind(error_type=type(exc).__name__).warning("crawler_browser_close_failed")


class _CrawlerCapability(CapabilityLifecycle[CrawlerWorkerResources]):
    """Crawler adapter for the generic process/loop capability owner."""

    @override
    async def provision(self) -> CrawlerWorkerResources:
        return await _provision_crawler_worker()

    @override
    async def release(self, resources: CrawlerWorkerResources) -> None:
        await _close_crawler_browser(resources)


_CRAWLER_OWNER = CapabilityOwner(_CrawlerCapability())


def _sync_crawler_compat_state() -> None:
    """Mirror owner state for older tests and external worker callers."""
    global _WORKER_PID, _WORKER_RUNNER, _WORKER_RESOURCES
    _WORKER_PID, _WORKER_RUNNER, _WORKER_RESOURCES = _CRAWLER_OWNER._state()


@worker_process_init.connect
def initialize_crawler_worker(**_kwargs: object) -> None:
    """Reset inherited state; first task provisions resources after child startup."""
    _CRAWLER_OWNER.reset_after_fork()
    _sync_crawler_compat_state()


def _ensure_crawler_worker() -> None:
    """Lazily provision once on the owning child's persistent loop."""
    try:
        _CRAWLER_OWNER.ensure()
    except CapabilityUsageError:
        raise
    except Exception as exc:
        exc.add_note("capability=crawler, operation=worker_provision")
        _sync_crawler_compat_state()
        raise ServiceUnavailableException(detail="Crawler worker provisioning failed") from exc
    else:
        _sync_crawler_compat_state()
        logger.info("Crawler worker resources initialized")


def get_crawler_worker_resources() -> CrawlerWorkerResources:
    """Return provisioned resources or a typed, capability-naming failure."""
    _ensure_crawler_worker()
    resources = _WORKER_RESOURCES
    if resources is None:
        raise ServiceUnavailableException(
            detail="Crawler worker is unavailable",
            data={"capability": "crawler_browser"},
        )
    return resources


def run_on_crawler_worker_loop[T](
    coroutine_factory: Callable[[], Coroutine[Any, Any, T]],
) -> T:
    """Run task I/O on the same loop that created the process resources.

    A soft time limit escapes ``run_until_complete`` without cancelling the
    coroutine, which would otherwise resume on this loop alongside the next
    task. Cancel and drain it before the next task runs.
    """
    get_crawler_worker_resources()
    result = _CRAWLER_OWNER.run(coroutine_factory)
    _sync_crawler_compat_state()
    return result


async def _probe_crawler_browser() -> bool:
    """Return True when the shared browser still executes a trivial load."""
    resources = _CRAWLER_OWNER.peek()
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
    if _CRAWLER_OWNER.peek() is None:
        return False
    try:
        return _CRAWLER_OWNER.run(_probe_crawler_browser)
    except CapabilityUsageError:
        raise
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        exc.add_note("operation=crawler_browser_probe")
        return False


async def aprobe_crawler_browser() -> bool:
    """Probe the shared browser from inside its owning loop."""
    if _CRAWLER_OWNER.peek() is None:
        return False
    return await _probe_crawler_browser()


async def areset_crawler_worker() -> None:
    """Discard the shared browser from inside its owning loop.

    Unlike :func:`reset_crawler_worker`, this must not touch the runner:
    the loop is running, so it can neither run a close coroutine through
    ``runner.run`` nor be closed. The next ``_ensure`` re-provisions the
    browser on the same runner.
    """
    await _CRAWLER_OWNER.reset_in_loop()
    _sync_crawler_compat_state()


def reset_crawler_worker() -> None:
    """Discard the shared browser and runner; the next task re-provisions.

    Sync contexts only (task boundary, shutdown): the loop must not be
    running. From inside the owning loop, use :func:`areset_crawler_worker`.
    """
    try:
        _CRAWLER_OWNER.shutdown()
    except CapabilityUsageError:
        raise
    except Exception as exc:  # noqa: BLE001 — shutdown must never raise
        exc.add_note("operation=reset_crawler_worker")
        logger.bind(error_type=type(exc).__name__).warning("crawler_browser_close_failed")
    finally:
        _sync_crawler_compat_state()


@worker_process_shutdown.connect
@worker_shutdown.connect
def shutdown_crawler_worker(**_kwargs: object) -> None:
    """Release the browser once and close the persistent loop."""
    if _CRAWLER_OWNER.peek() is None:
        logger.info("Crawler worker had no resources to close")
    try:
        _CRAWLER_OWNER.shutdown()
        logger.info("Crawler worker resources closed")
    except CapabilityUsageError:
        raise
    except Exception as exc:  # noqa: BLE001 — shutdown must never raise
        exc.add_note("operation=shutdown_crawler_worker")
        logger.bind(error_type=type(exc).__name__).warning("crawler_browser_close_failed")
    finally:
        _sync_crawler_compat_state()
