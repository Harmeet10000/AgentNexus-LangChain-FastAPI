"""Regression tests for crawler lifecycle and bounded content behavior."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.features.crawler.dto import CrawlRequest
from app.features.crawler.service import CrawlerService
from app.shared.crawler.chunker import clean_markdown, smart_chunk_markdown, truncate_content
from app.shared.crawler.config import CrawlerConfig
from app.shared.crawler.crawler import CrawlResult, WebCrawler
from app.shared.crawler.processor import GeminiProcessor
from app.shared.crawler.validator import sanitize_url, validate_url, validate_url_for_fetch


class _FakeCrawler:
    async def crawl(self, **_kwargs: object) -> object:
        return SimpleNamespace(
            unwrap=lambda: CrawlResult(
                url="https://example.com",
                success=True,
                markdown="# Title\n\nContent",
                links=[],
            )
        )


@pytest.mark.asyncio
async def test_service_uses_injected_crawler_without_nested_event_loop() -> None:
    service = CrawlerService(crawler=_FakeCrawler())  # type: ignore[arg-type]

    result = await service.crawl(CrawlRequest(url="https://example.com"))

    assert result.unwrap().successful_pages == 1


def test_truncate_content_reports_bounded_output() -> None:
    result = truncate_content("abcdefghij", max_length=5)

    assert len(result) <= 5


def test_chunker_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="max_len"):
        smart_chunk_markdown("content", max_len=0)


def test_chunker_returns_content_without_headings() -> None:
    chunks = smart_chunk_markdown("content without a markdown heading", max_len=10)

    assert chunks
    assert "content" in "".join(chunk.text for chunk in chunks)


def test_chunker_preserves_heading_context_for_split_sections() -> None:
    chunks = smart_chunk_markdown("# Root\n\n## Child\n\n" + "word " * 100, max_len=50)

    assert len(chunks) > 1
    assert all(chunk.headers for chunk in chunks)


def test_chunker_ignores_headings_inside_code_and_preserves_code_whitespace() -> None:
    markdown = "# Real\n\n```python\n# Not a heading\n\nvalue = 1\n```\n\nBody"

    chunks = smart_chunk_markdown(markdown, max_len=200)

    assert len(chunks) == 1
    assert chunks[0].headers == "# Real"
    assert clean_markdown("```\nline\n\n\nline\n```") == "```\nline\n\n\nline\n```"


def test_chunker_supports_overlap_and_setext_headings() -> None:
    chunks = smart_chunk_markdown("Title\n=====\n\n" + ("word " * 30), max_len=40, overlap=5)

    assert len(chunks) > 1
    assert chunks[0].headers == "# Title"
    assert any(chunks[index].text[:5] in chunks[index - 1].text for index in range(1, len(chunks)))


def test_web_crawler_accepts_a_shared_browser() -> None:
    browser = object()

    crawler = WebCrawler(browser=browser)  # type: ignore[arg-type]

    assert crawler.browser is browser


def test_crawl_request_rejects_unknown_fields_and_bounds_agent_output() -> None:
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", unexpected=True)

    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", max_output_chars=0)


def test_crawl_request_supports_explicit_chunk_output() -> None:
    request = CrawlRequest(
        url="https://example.com",
        include_chunks=True,
        chunk_size=512,
        max_chunks=4,
    )

    assert request.include_chunks is True
    assert request.chunk_size == 512


def test_structured_extraction_requires_a_matching_schema() -> None:
    with pytest.raises(ValidationError):
        CrawlRequest(url="https://example.com", extract_structured=True)

    with pytest.raises(ValidationError):
        CrawlRequest(
            url="https://example.com",
            extract_structured=True,
            schema_type="custom",
        )


def test_crawl4ai_fit_markdown_is_preferred_when_available() -> None:
    markdown = SimpleNamespace(raw_markdown="raw", fit_markdown="filtered")
    result = SimpleNamespace(
        success=True,
        url="https://example.com",
        markdown=markdown,
        html="<p>filtered</p>",
        metadata={"title": "Example"},
        links={"internal": []},
    )

    converted = WebCrawler._to_crawl_result(result, 0)

    assert converted.markdown == "filtered"


def test_process_content_limit_is_applied_before_cache() -> None:
    crawler = WebCrawler(config=CrawlerConfig(max_content_size=10))
    result = CrawlResult(
        url="https://example.com",
        success=True,
        markdown="a very long markdown document",
        html="<p>a very long html document</p>",
    )

    bounded = crawler._bound_result(result)

    assert len(bounded.markdown or "") <= 10
    assert len(bounded.html or "") <= 10


def test_cache_key_includes_content_policy() -> None:
    first = WebCrawler(config=CrawlerConfig(max_content_size=100))
    second = WebCrawler(config=CrawlerConfig(max_content_size=200))

    assert first._get_cache_key("https://example.com") != second._get_cache_key(
        "https://example.com"
    )


@pytest.mark.asyncio
async def test_recursive_crawl_preserves_partial_page_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Browser:
        async def arun_many(self, **_kwargs: object) -> list[object]:
            return [
                SimpleNamespace(
                    success=True,
                    url="https://example.com/ok",
                    markdown=None,
                    html=None,
                    metadata={},
                    links={},
                ),
                SimpleNamespace(
                    success=False,
                    url="https://example.com/fail",
                    error_message="timeout",
                ),
            ]

    async def allow_url(_url: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr("app.shared.crawler.crawler.validate_url_for_fetch", allow_url)
    crawler = WebCrawler(config=CrawlerConfig(), browser=_Browser())  # type: ignore[arg-type]

    result = await crawler.crawl_recursive(["https://example.com"])

    assert result.unwrap()[0].success is True
    assert result.unwrap()[1].success is False


@pytest.mark.asyncio
async def test_processor_uses_async_model_invocation() -> None:
    class _Model:
        def invoke(self, _prompt: str) -> object:
            message = "synchronous model invocation must not be used"
            raise AssertionError(message)

        async def ainvoke(self, _prompt: str) -> object:
            return SimpleNamespace(content="A short summary")

    result = await GeminiProcessor(model=_Model()).summarize("content")  # type: ignore[arg-type]

    assert result.success is True
    assert result.summary == "A short summary"


@pytest.mark.asyncio
async def test_html_mode_applies_the_output_budget_to_html() -> None:
    class _Crawler:
        async def crawl(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                unwrap=lambda: CrawlResult(
                    url="https://example.com",
                    success=True,
                    markdown="short",
                    html="<html>" + ("x" * 2_000) + "</html>",
                    links=[],
                )
            )

    service = CrawlerService(crawler=_Crawler())  # type: ignore[arg-type]
    result = await service.crawl(
        CrawlRequest(
            url="https://example.com",
            mode="html",
            max_output_chars=256,
            max_total_output_chars=1_024,
        )
    )

    item = result.unwrap().results[0]
    assert len(item.html or "") <= 256
    assert len(item.markdown or "") <= 256


def test_url_validation_rejects_embedded_credentials() -> None:
    valid, message = validate_url("https://user:password@example.com")

    assert valid is False
    assert "credentials" in message.lower()


def test_url_validation_normalizes_fragments_and_rejects_alternate_ports() -> None:
    assert sanitize_url("HTTPS://Example.COM/path#fragment") == "https://example.com/path"
    valid, message = validate_url("https://example.com:8080")
    assert valid is False
    assert "port" in message.lower()

    valid, message = validate_url("https://2130706433")
    assert valid is False
    assert "private" in message.lower()


@pytest.mark.asyncio
async def test_fetch_validation_rejects_private_dns_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(
        *_args: object, **_kwargs: object
    ) -> list[tuple[object, object, object, object, tuple[str, int]]]:
        return [(0, 0, 0, "", ("10.0.0.5", 443))]

    monkeypatch.setattr("app.shared.crawler.validator.socket.getaddrinfo", resolve)

    valid, message = await validate_url_for_fetch("https://public.example")

    assert valid is False
    assert "private" in message.lower()
