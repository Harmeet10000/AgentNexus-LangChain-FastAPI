"""Celery execution boundary for durable Crawl4AI jobs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import NoReturn

    from crawl4ai import AsyncWebCrawler
    from redis.asyncio import Redis

    from app.features.crawler.application.egress import CrawlerEgressPolicy

from celery.signals import worker_ready
from playwright.async_api import Error as PlaywrightError
from redis.exceptions import RedisError
from returns.result import Failure

from app.config import get_settings
from app.connections.redis import create_redis_client
from app.features.crawler.adapters.crawl4ai.crawler import WebCrawler
from app.features.crawler.adapters.persistence.redis.job_store import CrawlJobStore
from app.features.crawler.application.contracts import CrawlRequest
from app.features.crawler.application.egress import (
    build_egress_policy,
    is_boundary_outage,
    relay_redis_url,
)
from app.features.crawler.application.processor import get_processor
from app.features.crawler.application.service import CrawlerService
from app.observability.logging import logger
from app.shared.jobs.names import CRAWLER_CRAWL
from app.utils import InfrastructureException, ServiceUnavailableException
from app.worker.capabilities.crawler import (
    aprobe_crawler_browser,
    areset_crawler_worker,
    get_crawler_worker_resources,
    probe_crawler_browser,
    reset_crawler_worker,
    run_on_crawler_worker_loop,
)
from app.worker.celery.app import get_worker_celery_app
from app.worker.celery.task import ResilientTask

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

#: Bound for the boot-time reconciliation sweep (rows settled per boot).
_RECONCILE_MAX_ROWS = 100


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE_EXC)


def _resolve_egress_policy() -> CrawlerEgressPolicy:
    """Resolve the deployment's egress boundary for this worker process.

    The crawler worker has no route to public destinations and no route to its
    managed Redis other than the one this policy declares, so an absent or
    unsafe boundary fails the job closed instead of degrading it.
    """
    return build_egress_policy(get_settings())


def _create_worker_redis() -> Redis:
    """Create the crawl-execution Redis client, honouring the fixed relay.

    When the deployment declares a relay, the worker reaches its managed
    Redis *only* through it while a crawl runs. There is no direct fallback:
    a relay or TLS failure fails the job closed instead of reopening a public
    route. Job persistence is not crawl traffic — the terminal-write path uses
    :func:`_create_store_redis`, which never resolves the egress policy.
    """
    settings = get_settings()
    relay = _resolve_egress_policy().redis_relay
    if relay is None:
        return create_redis_client(settings.REDIS_URL)
    return create_redis_client(
        relay_redis_url(settings.REDIS_URL, relay),
        ssl_ca_certs=relay.ca_file,
        ssl_certfile=relay.cert_file,
        ssl_keyfile=relay.key_file,
    )


def _create_store_redis() -> Redis:
    """Create the job-store Redis client used for persistence writes.

    The egress policy gates the crawl, not persistence: a misconfigured or
    unavailable boundary must still let a terminal job state land instead of
    leaving zombie ``pending``/``cancelling`` rows behind. This is the same
    plain client every other worker builds via
    :func:`app.connections.redis.create_redis_client` — no egress-policy
    dependency — so the H1 misconfiguration class can only fail the crawl,
    never the failure write.
    """
    return create_redis_client(get_settings().REDIS_URL)


def _unhealthy_browser_failure(failure_message: str, crawl_id: str) -> NoReturn:
    """Raise a retryable failure for a browser-level crawl failure."""
    raise InfrastructureException(
        detail=f"Crawler browser unhealthy: {failure_message}",
        retryable=True,
        data={"crawl_id": crawl_id},
    )


def _boundary_outage_failure(crawl_id: str) -> NoReturn:
    """Raise a retryable failure for a transient egress-boundary outage."""
    raise InfrastructureException(
        detail="Crawler egress boundary is unavailable",
        retryable=True,
        data={"crawl_id": crawl_id},
    )


async def _mark_terminal(store: CrawlJobStore, crawl_id: str, message: str) -> None:
    """Write FAILED unless cancellation was requested, which wins over failure.

    Finalization consults the persisted cancellation request, so a ``cancelling``
    row with no running task (the H1 second zombie) settles to CANCELLED by the
    same non-gated write instead of being stranded.
    """
    try:
        if await store.is_cancel_requested(crawl_id):
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
    """Persist a terminal job status without the worker-owned browser loop.

    Uses the plain store client: this write must succeed even in the
    misconfigured-egress environment that killed the crawl itself.
    """
    redis = _create_store_redis()
    try:
        await _mark_terminal(CrawlJobStore(redis), crawl_id, message)
    except Exception as store_exc:  # noqa: BLE001 — best-effort terminal write
        store_exc.add_note(f"crawl_id={crawl_id}, operation=mark_failed_best_effort")
        logger.bind(
            crawl_id=crawl_id,
            error_type=type(store_exc).__name__,
            error=str(store_exc),
        ).exception("Could not persist crawler failure")
    finally:
        await redis.aclose(close_connection_pool=True)


def _mark_failed_sync(crawl_id: str, message: str) -> None:
    try:
        asyncio.run(_mark_failed_best_effort(crawl_id, message))
    except Exception as exc:  # noqa: BLE001 — terminal write must never mask retry
        exc.add_note(f"crawl_id={crawl_id}, operation=mark_failed_sync")
        logger.bind(
            crawl_id=crawl_id,
            error_type=type(exc).__name__,
            error=str(exc),
        ).exception("crawler_failure_write_failed")


async def _reconcile_orphaned_cancellations() -> int:
    """Settle ``cancelling`` rows left behind by the previous worker process.

    Runs once at boot, before this worker accepts any crawl: a CANCELLING row
    observed here was written by the dead process (this worker owns no tasks
    yet), so finalizing it to CANCELLED is truthful and safe — the persisted
    cancellation request wins, exactly as in :func:`_mark_terminal`. A task
    redelivered later for such a row would cancel itself through the same
    state, so the sweep cannot race a live writer into a lie.

    ``pending`` and ``running`` rows are deliberately left alone: their tasks
    may still sit queued for delivery (acks-late requeues worker-loss), and
    settling them could mark a job FAILED that then runs to COMPLETED.
    """
    redis = _create_store_redis()
    try:
        store = CrawlJobStore(redis)
        crawl_ids = await store.cancelling_crawl_ids(limit=_RECONCILE_MAX_ROWS)
        for crawl_id in crawl_ids:
            try:
                await store.mark_cancelled(crawl_id)
            except Exception as settle_exc:  # noqa: BLE001 — one stale row must not abort the sweep
                settle_exc.add_note(f"crawl_id={crawl_id}, operation=crawler_boot_reconcile")
                logger.bind(
                    crawl_id=crawl_id,
                    error_type=type(settle_exc).__name__,
                    error=str(settle_exc),
                ).warning("crawler_boot_reconcile_row_failed")
        return len(crawl_ids)
    finally:
        await redis.aclose(close_connection_pool=True)


@worker_ready.connect
def reconcile_orphaned_crawler_jobs(**_kwargs: object) -> None:
    """One-time boot reconciliation for the crawler worker's zombie rows."""
    try:
        settled = asyncio.run(_reconcile_orphaned_cancellations())
    except Exception as exc:  # noqa: BLE001 — reconciliation must never fail boot
        exc.add_note("operation=crawler_boot_reconcile")
        logger.bind(error_type=type(exc).__name__, error=str(exc)).exception(
            "crawler_boot_reconcile_failed"
        )
        return
    if settled:
        logger.bind(settled=settled).info("crawler_boot_reconcile_completed")


async def _run_crawl_job(
    crawl_id: str,
    request_payload: dict[str, object],
    *,
    final_attempt: bool,
    browser: AsyncWebCrawler,
) -> None:
    redis = _create_worker_redis()
    store = CrawlJobStore(redis)
    try:
        if await store.is_cancelled(crawl_id):
            await store.mark_cancelled(crawl_id)
            return
        await store.mark_running(crawl_id)
        request = CrawlRequest.model_validate(request_payload)
        # A provider transport that cannot honour the egress boundary stays
        # disabled: the job is refused with a bounded terminal outcome rather
        # than silently regaining direct egress.
        external_processing = _resolve_egress_policy().external_processing_enabled
        if not external_processing and (request.extract_structured or request.summary):
            await _mark_terminal(
                store,
                crawl_id,
                "Crawler external processing is disabled by the egress policy",
            )
            return
        # The browser was provisioned at the task boundary and handed in: a
        # synchronous owner operation from inside this coroutine (which runs ON
        # the owner loop) raises CapabilityUsageError, which is exactly the
        # deterministic "capability owner synchronous operations must run at a
        # task boundary" failure the 2026-11-15 smoke recorded (F8 — every
        # crawl died in ~226 ms before attempting a page).
        processor = await get_processor() if external_processing else None
        crawler = WebCrawler(redis_client=redis, browser=browser)
        service = CrawlerService(crawler=crawler, processor=processor, redis_client=redis)
        result = await service.crawl(request, crawl_id=crawl_id)
        if await store.is_cancelled(crawl_id):
            await store.mark_cancelled(crawl_id)
            return
        if isinstance(result, Failure):
            failure_message = result.failure().message
            if is_boundary_outage(failure_message):
                # A proxy outage or TLS failure is transient: keep the job
                # retryable instead of terminating it as a content failure.
                _boundary_outage_failure(crawl_id)
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
        # Provision (or reuse) the browser at the task boundary — the loop is
        # not running here — and hand the resource into the coroutine. Keep
        # provisioning inside the terminal-state guard so a persistent policy
        # outage cannot strand an accepted job in PENDING.
        browser = get_crawler_worker_resources().browser
        run_on_crawler_worker_loop(
            lambda: _run_crawl_job(
                crawl_id, request_payload, final_attempt=final_attempt, browser=browser
            )
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
