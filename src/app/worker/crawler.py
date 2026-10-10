"""Per-child crawler resources, provisioned on first task use."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from celery.signals import worker_process_init, worker_process_shutdown, worker_shutdown
from crawl4ai import AsyncWebCrawler  # noqa: TC002 — resolved at runtime by Pydantic
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
    """Run task I/O on the same loop that created the process resources."""
    get_crawler_worker_resources()
    if _WORKER_RUNNER is None:
        message = "Worker resource guard returned without a runner"
        raise AssertionError(message)
    return _WORKER_RUNNER.run(coroutine_factory())


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
        runner.run(close_crawl4ai_crawler(resources.browser))
        logger.info("Crawler worker resources closed")
    except (OSError, RuntimeError) as exc:
        exc.add_note("operation=shutdown_crawler_worker")
        logger.exception("Could not close per-worker crawler browser")
    finally:
        if runner is not None:
            runner.close()
