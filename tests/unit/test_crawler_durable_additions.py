"""Durable additions: browser network policy, worker reuse, job tools."""

import sys
from types import SimpleNamespace

import pytest
from fakeredis.aioredis import FakeRedis

from app.features.crawler.dto import CrawlChunk, CrawlResponse, CrawlResultItem
from app.features.crawler.job_store import CrawlJobStore
from app.shared.crawler import WebCrawler
from app.shared.crawler.validator import validate_browser_result_urls
from app.shared.langchain_layer.agents.tools.crawl_jobs import get_crawl_job_tools


def test_browser_result_urls_reject_non_http_schemes() -> None:
    result = SimpleNamespace(url="https://example.com", redirected_url="file:///etc/passwd")
    allowed, _ = validate_browser_result_urls(result)
    assert allowed is False

    clean = SimpleNamespace(url="https://example.com/page", redirected_url=None)
    allowed, _ = validate_browser_result_urls(clean)
    assert allowed is True


def test_cache_key_is_isolated_by_crawler_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.shared.crawler.crawler as crawler_module

    monkeypatch.setattr(crawler_module, "_crawl4ai_version", lambda: "1.0.0")
    first = WebCrawler()._get_cache_key("https://example.com")
    monkeypatch.setattr(crawler_module, "_crawl4ai_version", lambda: "2.0.0")
    second = WebCrawler()._get_cache_key("https://example.com")
    assert first.startswith("crawl:cache:")
    assert first != second


@pytest.mark.asyncio
async def test_finish_single_result_rejects_private_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def deny(_url: str, *, phase: str) -> tuple[bool, str]:
        _ = phase
        if "evil-rebind.example" in _url:
            return False, "Hostname resolves to a private address: 10.0.0.5"
        return True, ""

    monkeypatch.setattr("app.shared.crawler.crawler.validate_navigation_destination", deny)
    raw = SimpleNamespace(
        success=True,
        url="https://evil-rebind.example/admin",
        redirected_url=None,
        markdown=SimpleNamespace(fit_markdown="x", raw_markdown="x"),
        html=None,
        metadata={},
        links={},
    )
    crawler = WebCrawler()
    outcome = await crawler._finish_single_result("https://example.com", raw, 0.0)
    assert isinstance(outcome.failure(), object)


@pytest.mark.asyncio
async def test_worker_browser_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    # tests/conftest.py intentionally masks `tasks` with a MagicMock (heavy
    # Celery imports), so restore the real package for this import only.
    real_tasks = sys.modules.pop("tasks", None)
    try:
        import tasks.crawler_tasks as worker_tasks
    finally:
        sys.modules.pop("tasks.crawler_tasks", None)
        sys.modules.pop("tasks", None)
        if real_tasks is not None:
            sys.modules["tasks"] = real_tasks

    calls = {"count": 0}

    async def fake_create() -> object:
        calls["count"] += 1
        return SimpleNamespace()

    monkeypatch.setattr(worker_tasks, "create_crawl4ai_crawler", fake_create)
    monkeypatch.setattr(worker_tasks, "_WORKER_BROWSER", None)
    first = await worker_tasks._get_shared_browser()
    second = await worker_tasks._get_shared_browser()
    assert first is second
    assert calls["count"] == 1
    monkeypatch.setattr(worker_tasks, "_WORKER_BROWSER", None)


@pytest.mark.asyncio
async def test_job_tools_return_bounded_agent_output() -> None:
    redis = FakeRedis(decode_responses=True)
    store = CrawlJobStore(redis, ttl_seconds=300)
    tools = {tool.name: tool for tool in get_crawl_job_tools(job_store=store)}
    assert {
        "crawl_start",
        "crawl_status",
        "crawl_get_page",
        "crawl_get_chunk",
        "crawl_search_chunks",
        "crawl_cancel",
    } <= set(tools)

    # Tools default to the agent owner namespace.
    start = await tools["crawl_start"]._arun(url="https://example.com")
    assert "crawl_id=" in start

    crawl_id = start.split("crawl_id=")[1].split()[0]
    status = await tools["crawl_status"]._arun(crawl_id=crawl_id)
    assert "status=pending" in status

    response = CrawlResponse(
        success=True,
        crawl_id=crawl_id,
        query_url="https://example.com",
        results=[
            CrawlResultItem(
                url="https://example.com",
                page_id="page-1",
                success=True,
                markdown="hello",
                chunks=[
                    CrawlChunk(
                        text="hello world", index=0, headers="h", char_count=11, word_count=2
                    )
                ],
            )
        ],
        total_pages=1,
        successful_pages=1,
        failed_pages=0,
        total_word_count=2,
        processing_time_ms=1,
    )
    await store.mark_running(crawl_id)
    await store.save_result(crawl_id, response)

    pages = await tools["crawl_get_page"]._arun(crawl_id=crawl_id)
    assert "page_id=page-1" in pages
    chunks = await tools["crawl_get_chunk"]._arun(crawl_id=crawl_id)
    assert "hello world" in chunks
    found = await tools["crawl_search_chunks"]._arun(crawl_id=crawl_id, query="hello")
    assert "matches=1" in found
    cancelled = await tools["crawl_cancel"]._arun(crawl_id=crawl_id)
    assert "crawl_id=" in cancelled
