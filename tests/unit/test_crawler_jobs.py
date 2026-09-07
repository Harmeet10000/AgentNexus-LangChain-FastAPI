"""Durable crawler-job lifecycle and retrieval contract tests."""

import pytest
from fakeredis.aioredis import FakeRedis

from app.features.crawler.dto import (
    CrawlChunk,
    CrawlJobStatus,
    CrawlRequest,
    CrawlResponse,
    CrawlResultItem,
)
from app.features.crawler.job_store import CrawlJobStore

pytestmark = pytest.mark.unit


def _response() -> CrawlResponse:
    return CrawlResponse(
        success=True,
        crawl_id="crawl-1",
        query_url="https://example.com",
        results=[
            CrawlResultItem(
                url="https://example.com",
                page_id="page-1",
                success=True,
                markdown="Durable crawler content",
                chunks=[
                    CrawlChunk(
                        text="Durable crawler content",
                        index=0,
                        headers="",
                        char_count=24,
                        word_count=3,
                    )
                ],
            )
        ],
        total_pages=1,
        successful_pages=1,
        failed_pages=0,
        total_word_count=3,
        processing_time_ms=10,
    )


@pytest.mark.asyncio
async def test_job_creation_is_idempotent_and_owner_scoped() -> None:
    redis = FakeRedis(decode_responses=True)
    store = CrawlJobStore(redis, ttl_seconds=300)
    payload = CrawlRequest(url="https://example.com").model_dump(mode="json")

    first, created = await store.create(
        crawl_id="crawl-1",
        owner="client-1",
        request_payload=payload,
        idempotency_key="same-request",
    )
    second, duplicate = await store.create(
        crawl_id="crawl-2",
        owner="client-1",
        request_payload=payload,
        idempotency_key="same-request",
    )

    assert created is True
    assert duplicate is False
    assert second.crawl_id == first.crawl_id
    assert await store.get("crawl-1", owner="other-client") is None


@pytest.mark.asyncio
async def test_result_is_persisted_and_paginated_as_pages_and_chunks() -> None:
    redis = FakeRedis(decode_responses=True)
    store = CrawlJobStore(redis, ttl_seconds=300)
    payload = CrawlRequest(url="https://example.com").model_dump(mode="json")
    await store.create(crawl_id="crawl-1", owner="client-1", request_payload=payload)

    await store.mark_running("crawl-1")
    job = await store.save_result("crawl-1", _response())
    pages = await store.pages("crawl-1", owner="client-1", cursor=0, limit=1)
    chunks = await store.chunks("crawl-1", owner="client-1", cursor=0, limit=1)
    found = await store.search_chunks("crawl-1", owner="client-1", query="durable", limit=5)

    assert job.status is CrawlJobStatus.COMPLETED
    assert pages is not None
    assert len(pages.items) == 1
    assert chunks is not None
    assert len(chunks[0]) == 1
    assert found is not None
    assert len(found.items) == 1


@pytest.mark.asyncio
async def test_cancel_sets_cooperative_flag_and_status() -> None:
    redis = FakeRedis(decode_responses=True)
    store = CrawlJobStore(redis, ttl_seconds=300)
    payload = CrawlRequest(url="https://example.com").model_dump(mode="json")
    await store.create(crawl_id="crawl-1", owner="client-1", request_payload=payload)

    job = await store.request_cancel("crawl-1", owner="client-1")

    assert job is not None
    assert job.status is CrawlJobStatus.CANCELLING
    assert await store.is_cancelled("crawl-1") is True
