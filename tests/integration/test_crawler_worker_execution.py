"""Deterministic worker-path coverage for durable crawler execution."""

from __future__ import annotations

import asyncio
import json
import subprocess  # noqa: S404 — trusted local interpreter bootstrap probe
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from crawl4ai import CrawlerRunConfig
from crawl4ai.async_configs import GeolocationConfig
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from loguru import logger as loguru_logger
from redis.exceptions import RedisError

from app.features.crawler.adapters.crawl4ai.config import CrawlerConfig
from app.features.crawler.adapters.inbound.worker import tasks as crawler_job
from app.features.crawler.adapters.persistence.redis.job_store import CrawlJobStore
from app.features.crawler.api.http.dto import CrawlRequest
from app.features.crawler.application.egress import CrawlerEgressConfigError
from app.features.crawler.domain.models import CrawlJobStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, NoReturn


class _ControlledBrowser:
    """Capture Crawl4AI configuration and return one controlled page outcome."""

    def __init__(self, result_factory: Callable[[str], object]) -> None:
        self._result_factory = result_factory
        self.seen_configs: list[CrawlerRunConfig] = []

    async def arun(self, *, url: str, config: CrawlerRunConfig) -> object:
        self.seen_configs.append(config)
        return self._result_factory(url)


def _successful_page(url: str) -> object:
    markdown = "# Controlled page\n\nDurable crawler content"
    return SimpleNamespace(
        success=True,
        url=url,
        redirected_url=None,
        markdown=SimpleNamespace(fit_markdown=markdown, raw_markdown=markdown),
        html="<h1>Controlled page</h1><p>Durable crawler content</p>",
        metadata={"title": "Controlled page"},
        links={"internal": []},
    )


def _failed_page(url: str) -> object:
    return SimpleNamespace(
        success=False,
        url=url,
        redirected_url=None,
        error_message="controlled target execution failed",
        links={},
    )


async def _allow_seed(_url: str) -> tuple[bool, str]:
    return True, ""


async def _allow_navigation(_url: str, *, phase: str) -> tuple[bool, str]:
    _ = phase
    return True, ""


async def _no_processor() -> None:
    return None


async def _healthy_browser() -> bool:
    return True


def _raise_misconfigured_egress() -> NoReturn:
    """Simulate the smoke's misconfiguration: CRAWLER_EGRESS_PROXY_URL unset."""
    message = "CRAWLER_EGRESS_PROXY_URL must be configured"
    raise CrawlerEgressConfigError(message)


def _capture_log_records(event_names: set[str]) -> tuple[list[dict[str, Any]], ExitStack]:
    """Capture full loguru records for the given event names."""
    records: list[dict[str, Any]] = []
    stack = ExitStack()

    def _sink(message: Any) -> None:
        record = message.record
        if record["message"] in event_names:
            records.append(record)

    sink_id = loguru_logger.add(_sink, level="DEBUG", format="{message}")
    stack.callback(loguru_logger.remove, sink_id)
    return records, stack


async def _arrange_job(
    monkeypatch: pytest.MonkeyPatch,
    *,
    crawl_id: str,
) -> tuple[CrawlJobStore, dict[str, object], FakeRedis]:
    fake_server = FakeServer()
    inspection_redis = FakeRedis(server=fake_server, decode_responses=True)
    worker_redis = FakeRedis(server=fake_server, decode_responses=True)
    store = CrawlJobStore(inspection_redis, ttl_seconds=300)
    request = CrawlRequest(url="https://example.com", bypass_cache=True)
    payload = request.model_dump(mode="json")
    await store.create(crawl_id=crawl_id, owner="client-1", request_payload=payload)

    monkeypatch.setattr(crawler_job, "create_redis_client", lambda _url, **_kwargs: worker_redis)
    monkeypatch.setattr(crawler_job, "get_processor", _no_processor)
    monkeypatch.setattr(
        "app.features.crawler.adapters.crawl4ai.crawler.get_crawler_config",
        lambda: CrawlerConfig(geolocation_lat=28.6139, geolocation_lon=77.2090),
    )
    monkeypatch.setattr(
        "app.features.crawler.adapters.crawl4ai.crawler.validate_url_for_fetch", _allow_seed
    )
    monkeypatch.setattr(
        "app.features.crawler.adapters.crawl4ai.crawler.validate_navigation_destination",
        _allow_navigation,
    )
    return store, payload, inspection_redis


@pytest.mark.asyncio
async def test_worker_persists_page_after_real_run_config_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawl_id = "crawl-worker-success"
    store, payload, inspection_redis = await _arrange_job(monkeypatch, crawl_id=crawl_id)
    browser = _ControlledBrowser(_successful_page)

    await crawler_job._run_crawl_job(
        crawl_id,
        payload,
        final_attempt=True,
        browser=browser,  # type: ignore[arg-type] — controlled Crawl4AI adapter
    )

    job = await store.get(crawl_id, owner="client-1")
    result = await store.get_result(crawl_id, owner="client-1")
    pages = await store.pages(crawl_id, owner="client-1", cursor=0, limit=10)

    assert len(browser.seen_configs) == 1
    run_config = browser.seen_configs[0]
    assert isinstance(run_config, CrawlerRunConfig)
    assert isinstance(run_config.geolocation, GeolocationConfig)
    assert job is not None
    assert job.status is CrawlJobStatus.COMPLETED
    assert result is not None
    assert result.successful_pages == 1
    assert pages is not None
    assert len(pages.items) == 1
    assert pages.items[0].title == "Controlled page"
    assert pages.items[0].markdown == "# Controlled page\n\nDurable crawler content"
    await inspection_redis.aclose(close_connection_pool=True)


@pytest.mark.asyncio
async def test_worker_records_target_failure_after_valid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crawl_id = "crawl-worker-target-failure"
    store, payload, inspection_redis = await _arrange_job(monkeypatch, crawl_id=crawl_id)
    browser = _ControlledBrowser(_failed_page)
    monkeypatch.setattr(crawler_job, "aprobe_crawler_browser", _healthy_browser)

    await crawler_job._run_crawl_job(
        crawl_id,
        payload,
        final_attempt=True,
        browser=browser,  # type: ignore[arg-type] — controlled Crawl4AI adapter
    )

    job = await store.get(crawl_id, owner="client-1")

    assert len(browser.seen_configs) == 1
    assert isinstance(browser.seen_configs[0], CrawlerRunConfig)
    assert job is not None
    assert job.status is CrawlJobStatus.FAILED
    assert job.error_message == "controlled target execution failed"
    assert "geolocation" not in job.error_message.lower()
    await inspection_redis.aclose(close_connection_pool=True)


@pytest.mark.asyncio
async def test_terminal_failure_write_survives_misconfigured_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1 (2026-11-19): the egress policy gates the crawl, not persistence.

    With egress-policy resolution raising (the unset CRAWLER_EGRESS_PROXY_URL
    misconfiguration), the task-boundary terminal write must still settle the
    job to FAILED through the plain job-store client instead of leaving a
    zombie ``pending`` row behind.
    """
    crawl_id = "crawl-egress-misconfig-failed"
    store, _payload, inspection_redis = await _arrange_job(monkeypatch, crawl_id=crawl_id)
    monkeypatch.setattr(crawler_job, "_resolve_egress_policy", _raise_misconfigured_egress)

    await asyncio.to_thread(
        crawler_job._mark_failed_sync,
        crawl_id,
        "Crawler egress boundary is misconfigured: CRAWLER_EGRESS_PROXY_URL must be configured",
    )

    job = await store.get(crawl_id, owner="client-1")
    assert job is not None
    assert job.status is CrawlJobStatus.FAILED
    assert job.error_message.startswith("Crawler egress boundary is misconfigured")
    await inspection_redis.aclose(close_connection_pool=True)


@pytest.mark.asyncio
async def test_cancelling_job_without_running_task_finalizes_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H1 (2026-11-19): a ``cancelling`` row with no running task must settle.

    The API-side cancel flips a stranded job to ``cancelling``; the worker-side
    finalize must still write CANCELLED through the non-gated client even when
    the egress policy is misconfigured.
    """
    crawl_id = "crawl-egress-misconfig-cancelled"
    store, _payload, inspection_redis = await _arrange_job(monkeypatch, crawl_id=crawl_id)
    await store.request_cancel(crawl_id, owner="client-1")
    monkeypatch.setattr(crawler_job, "_resolve_egress_policy", _raise_misconfigured_egress)

    await asyncio.to_thread(
        crawler_job._mark_failed_sync, crawl_id, "Crawler egress boundary is misconfigured"
    )

    job = await store.get(crawl_id, owner="client-1")
    assert job is not None
    assert job.status is CrawlJobStatus.CANCELLED
    await inspection_redis.aclose(close_connection_pool=True)


@pytest.mark.asyncio
async def test_cancelling_state_finalizes_even_after_flag_expiry() -> None:
    """The persisted ``cancelling`` state alone must request cancellation.

    A terminal write extends the record's TTL while the cancel flag keeps its
    original countdown, so finalize must honour the row state, not just the
    flag — otherwise a cancelled job settles to FAILED once the flag expires.
    """
    crawl_id = "crawl-cancel-flag-expired"
    store = CrawlJobStore(FakeRedis(decode_responses=True), ttl_seconds=300)
    await store.create(
        crawl_id=crawl_id, owner="client-1", request_payload={"url": "https://example.com"}
    )
    await store.request_cancel(crawl_id, owner="client-1")
    assert await store.is_cancel_requested(crawl_id) is True

    await store.redis.delete(CrawlJobStore._cancel_key(crawl_id))

    assert await store.is_cancelled(crawl_id) is False
    assert await store.is_cancel_requested(crawl_id) is True


def test_terminal_write_failure_logs_the_real_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H2 (2026-11-19): ``crawler_failure_write_failed`` carries the exception.

    The smoke recorded ``exception: null`` because the client construction
    raised outside every except-logging handler; the log line must always
    carry the exception type, message, and add_note chain.
    """

    def broken_store_client() -> NoReturn:
        message = "worker redis url unset"
        raise RuntimeError(message)

    monkeypatch.setattr(crawler_job, "_create_store_redis", broken_store_client)

    records, stack = _capture_log_records({"crawler_failure_write_failed"})
    with stack:
        crawler_job._mark_failed_sync("crawl-write-log-check", "egress misconfigured")

    assert len(records) == 1
    record = records[0]
    assert record["extra"]["error_type"] == "RuntimeError"
    assert "worker redis url unset" in str(record["extra"]["error"])
    exception = record["exception"]
    assert exception is not None
    assert exception[0] is RuntimeError
    notes = list(getattr(exception[1], "__notes__", []))
    assert any("operation=mark_failed_sync" in note for note in notes)


async def _seed_reconciliation_rows() -> tuple[dict[str, str], FakeRedis, FakeServer]:
    """Seed one row per lifecycle state on a fresh fakeredis server."""
    fake_server = FakeServer()
    inspection = FakeRedis(server=fake_server, decode_responses=True)
    store = CrawlJobStore(inspection, ttl_seconds=300)
    states: dict[str, str] = {}

    async def _create(crawl_id: str) -> None:
        await store.create(
            crawl_id=crawl_id, owner="client-1", request_payload={"url": "https://example.com"}
        )

    await _create("crawl-recon-pending")
    states["crawl-recon-pending"] = "pending"

    await _create("crawl-recon-running")
    await store.mark_running("crawl-recon-running")
    states["crawl-recon-running"] = "running"

    await _create("crawl-recon-failed")
    await store.mark_failed("crawl-recon-failed", "stale failure")
    states["crawl-recon-failed"] = "failed"

    await _create("crawl-recon-cancelled")
    await store.mark_cancelled("crawl-recon-cancelled")
    states["crawl-recon-cancelled"] = "cancelled"

    await _create("crawl-recon-cancelling")
    await store.request_cancel("crawl-recon-cancelling", owner="client-1")
    states["crawl-recon-cancelling"] = "cancelling"
    return states, inspection, fake_server


async def _live_status(redis: FakeRedis, crawl_id: str) -> str:
    raw = await redis.get(CrawlJobStore._meta_key(crawl_id))
    assert raw is not None
    payload: dict[str, Any] = json.loads(raw)
    return str(payload["status"])


@pytest.mark.asyncio
async def test_boot_reconcile_settles_only_orphaned_cancellations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boot reconciliation finalizes cancelling rows and nothing else.

    The 2026-11-19 sweep left ``288a07df`` stuck in CANCELLING with no running
    task: the finalize only ever ran inside a crawl task, so a row whose task
    died was unreachable. At boot this worker owns no tasks, so a CANCELLING
    row is orphaned by construction; ``pending``/``running`` rows are left
    alone because their tasks may still be queued for delivery.
    """
    states, inspection, fake_server = await _seed_reconciliation_rows()
    worker_redis = FakeRedis(server=fake_server, decode_responses=True)
    monkeypatch.setattr(crawler_job, "create_redis_client", lambda _url, **_kwargs: worker_redis)

    settled = await crawler_job._reconcile_orphaned_cancellations()

    assert settled == 1
    assert await _live_status(inspection, "crawl-recon-cancelling") == "cancelled"
    for crawl_id, state in states.items():
        if crawl_id != "crawl-recon-cancelling":
            assert await _live_status(inspection, crawl_id) == state

    # Idempotent: a second boot finds nothing left to settle.
    assert await crawler_job._reconcile_orphaned_cancellations() == 0
    await inspection.aclose(close_connection_pool=True)


@pytest.mark.asyncio
async def test_boot_reconcile_is_bounded_by_the_row_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep settles at most the configured number of rows per boot."""
    _states, inspection, fake_server = await _seed_reconciliation_rows()
    extra = FakeRedis(server=fake_server, decode_responses=True)
    extra_store = CrawlJobStore(extra, ttl_seconds=300)
    await extra_store.create(
        crawl_id="crawl-recon-cancelling-2",
        owner="client-1",
        request_payload={"url": "https://example.com"},
    )
    await extra_store.request_cancel("crawl-recon-cancelling-2", owner="client-1")
    await extra.aclose(close_connection_pool=True)

    worker_redis = FakeRedis(server=fake_server, decode_responses=True)
    monkeypatch.setattr(crawler_job, "create_redis_client", lambda _url, **_kwargs: worker_redis)
    monkeypatch.setattr(crawler_job, "_RECONCILE_MAX_ROWS", 1)

    settled = await crawler_job._reconcile_orphaned_cancellations()

    assert settled == 1
    settled_rows = [
        crawl_id
        for crawl_id in ("crawl-recon-cancelling", "crawl-recon-cancelling-2")
        if await _live_status(inspection, crawl_id) == "cancelled"
    ]
    assert len(settled_rows) == 1
    await inspection.aclose(close_connection_pool=True)


def test_boot_reconcile_failure_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reconciliation outage must not fail worker boot."""
    records, stack = _capture_log_records({"crawler_boot_reconcile_failed"})

    async def unreachable() -> int:
        message = "redis unreachable at boot"
        raise RedisError(message)

    monkeypatch.setattr(crawler_job, "_reconcile_orphaned_cancellations", unreachable)

    with stack:
        crawler_job.reconcile_orphaned_crawler_jobs()

    assert len(records) == 1
    record = records[0]
    assert record["extra"]["error_type"] == "RedisError"
    assert "redis unreachable at boot" in str(record["extra"]["error"])


_CRAWLER_BOOT_REGISTRATION_SCRIPT = """
import sys
from unittest.mock import patch

from celery.signals import worker_ready

from app.worker.celery.app import CatalogValidatingWorker, create_worker_celery_app
from app.worker.celery.profiles import profile_for_role

profile = profile_for_role("crawler")
app = create_worker_celery_app("crawler")
assert "app.features.crawler.adapters.inbound.worker.tasks" in (app.conf.include or ()), app.conf.include

# Worker construction must import the role's task modules through celery's
# include machinery (loader.init_worker), which is where the boot reconcile
# connects itself to worker_ready — before the consumer dispatches ready.
worker = CatalogValidatingWorker(
    app=app,
    queues=[profile.queue],
    pool=profile.pool,
    concurrency=1,
    hostname="crawler-recon-probe@probe",
    loglevel="info",
)
assert "app.features.crawler.adapters.inbound.worker.tasks" in sys.modules

import app.features.crawler.adapters.inbound.worker.tasks as crawler_jobs  # noqa: PLC0415

fired: list[str] = []


async def _stub() -> int:
    fired.append("handler")
    return 0


with patch.object(crawler_jobs, "_reconcile_orphaned_cancellations", _stub):
    worker_ready.send(sender=object())

assert fired == ["handler"], "worker_ready dispatch did not reach the boot reconcile"
print("registration-ok")
"""


def test_boot_reconcile_is_registered_on_crawler_worker_construction() -> None:
    """The reconcile must ride the real bootstrap, not a direct invocation.

    The 2026-11-20 hand-back: unit tests invoked the hook directly and passed
    while the live boot was declared silent. This pins the actual chain in a
    fresh process — the crawler role's ``include`` lists the task module,
    constructing the role worker imports it (connecting the ``worker_ready``
    receiver), and a ready dispatch reaches the handler.
    """
    result = subprocess.run(  # noqa: S603 — literal script with trusted interpreter
        [sys.executable, "-c", _CRAWLER_BOOT_REGISTRATION_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "registration-ok" in result.stdout
