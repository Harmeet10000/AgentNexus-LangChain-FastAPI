"""Crawler route tests for the lifespan-owned JobBus publication seam."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request, Response

from app.features.crawler import router as crawler_router
from app.features.crawler.dto import CrawlJob, CrawlJobStatus, CrawlRequest
from app.shared.jobs import DirectEnqueueReceipt
from app.shared.jobs.names import CRAWLER_CRAWL
from app.utils import ServiceUnavailableException

pytestmark = pytest.mark.unit


def _request(app: FastAPI) -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "client": ("127.0.0.1", 8000),
        }
    )


def _job(crawl_id: str = "crawl-1") -> CrawlJob:
    return CrawlJob(
        crawl_id=crawl_id,
        status=CrawlJobStatus.PENDING,
        created_at=datetime.now(UTC),
    )


class _RateLimiter:
    async def acquire(self, _owner: str, _scope: object) -> tuple[bool, dict[str, object]]:
        return True, {}


class _Store:
    def __init__(self, *, created: bool = True) -> None:
        self.created = created
        self.job = _job()
        self.created_args: dict[str, object] = {}
        self.attached: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []

    async def create(self, **kwargs: object) -> tuple[CrawlJob, bool]:
        self.created_args = kwargs
        return self.job, self.created

    async def attach_task_id(self, crawl_id: str, task_id: str) -> None:
        self.attached.append((crawl_id, task_id))

    async def mark_failed(self, crawl_id: str, message: str) -> CrawlJob:
        self.failed.append((crawl_id, message))
        return self.job


class _Bus:
    def __init__(self, *, task_id: str = "task-1", error: Exception | None = None) -> None:
        self.task_id = task_id
        self.error = error
        self.jobs: list[object] = []

    async def enqueue(self, job: object) -> DirectEnqueueReceipt:
        self.jobs.append(job)
        if self.error is not None:
            raise self.error
        return DirectEnqueueReceipt(task_id=self.task_id)


@pytest.mark.asyncio
async def test_create_route_uses_catalog_job_bus_and_persists_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler_router, "get_client_identifier", lambda _request: "owner-1")
    store = _Store()
    bus = _Bus(task_id="broker-1")
    app = FastAPI()
    response = Response()

    result = await crawler_router.crawl_url(
        CrawlRequest(url="https://example.com"),
        _request(app),
        response,
        _RateLimiter(),
        store,
        bus,
    )

    assert response.status_code == 202
    assert len(bus.jobs) == 1
    typed_job = bus.jobs[0]
    assert typed_job.task_name == CRAWLER_CRAWL
    assert typed_job.wire_payload()["crawl_id"] == "crawl-1"
    assert store.attached == [("crawl-1", "broker-1")]
    assert store.failed == []
    assert result.data == store.job


@pytest.mark.asyncio
async def test_create_route_marks_redis_job_failed_when_bus_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler_router, "get_client_identifier", lambda _request: "owner-1")
    store = _Store()

    with pytest.raises(ServiceUnavailableException, match="Durable crawler workers"):
        await crawler_router.crawl_url(
            CrawlRequest(url="https://example.com"),
            _request(FastAPI()),
            Response(),
            _RateLimiter(),
            store,
            None,
        )

    assert store.failed == [("crawl-1", "Celery is unavailable")]
    assert store.attached == []


@pytest.mark.asyncio
async def test_create_route_compensates_when_broker_publication_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler_router, "get_client_identifier", lambda _request: "owner-1")
    store = _Store()
    bus = _Bus(error=ServiceUnavailableException("broker unavailable"))

    with pytest.raises(ServiceUnavailableException, match="Could not enqueue crawler job"):
        await crawler_router.crawl_url(
            CrawlRequest(url="https://example.com"),
            _request(FastAPI()),
            Response(),
            _RateLimiter(),
            store,
            bus,
        )

    assert store.failed == [("crawl-1", "Could not enqueue crawler job")]
    assert store.attached == []


@pytest.mark.asyncio
async def test_duplicate_idempotent_create_skips_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler_router, "get_client_identifier", lambda _request: "owner-1")
    store = _Store(created=False)
    bus = _Bus()

    await crawler_router.crawl_url(
        CrawlRequest(url="https://example.com"),
        _request(FastAPI()),
        Response(),
        _RateLimiter(),
        store,
        bus,
    )

    assert bus.jobs == []
    assert store.attached == []
    assert store.failed == []


@pytest.mark.asyncio
async def test_cancel_route_still_revokes_raw_celery_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler_router, "get_client_identifier", lambda _request: "owner-1")
    revoke = Mock()
    app = FastAPI()
    app.state.celery = SimpleNamespace(control=SimpleNamespace(revoke=revoke))

    class CancelStore:
        async def request_cancel(self, _crawl_id: str, *, owner: str) -> CrawlJob:
            assert owner == "owner-1"
            return _job()

        async def task_id(self, _crawl_id: str, *, owner: str) -> str:
            assert owner == "owner-1"
            return "broker-1"

    await crawler_router.cancel_crawl(_request(app), "crawl-1", CancelStore())

    revoke.assert_called_once_with("broker-1", terminate=False)
