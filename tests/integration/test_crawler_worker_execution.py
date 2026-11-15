"""Deterministic worker-path coverage for durable crawler execution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from crawl4ai import CrawlerRunConfig
from crawl4ai.async_configs import GeolocationConfig
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from app.features.crawler.adapters.crawl4ai.config import CrawlerConfig
from app.features.crawler.adapters.persistence.redis.job_store import CrawlJobStore
from app.features.crawler.api.http.dto import CrawlRequest
from app.features.crawler.domain.models import CrawlJobStatus
from app.worker.jobs import crawler as crawler_job

if TYPE_CHECKING:
    from collections.abc import Callable


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

    monkeypatch.setattr(crawler_job, "create_redis_client", lambda _url: worker_redis)
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
