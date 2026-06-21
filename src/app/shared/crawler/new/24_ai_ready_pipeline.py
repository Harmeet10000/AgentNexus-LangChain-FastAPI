"""Video 24: Build an AI-ready Crawl4AI data pipeline.

Demonstrates:
- discover -> crawl -> filter -> export workflow
- deep crawl for discovery plus arun_many() for processing
- route-specific configs, rate limiting, and JSON/markdown export

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_24_ai_ready_pipeline.py`
"""

import asyncio
import json
import re
import time
from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler,
    BM25ContentFilter,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    JsonCssExtractionStrategy,
    LXMLWebScrapingStrategy,
    MatchMode,
    PruningContentFilter,
    RateLimiter,
)
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, FilterChain, URLPatternFilter

ROOT_URL = "https://docs.crawl4ai.com/"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "video_24"


def ensure_output_dir() -> tuple[Path, Path]:
    base = OUTPUT_DIR
    markdown_dir = base / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    return base, markdown_dir


def slugify(url: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")[:80] or "page"


async def discover_urls() -> list[str]:
    config = CrawlerRunConfig(
        check_robots_txt=True,
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=1,
            max_pages=8,
            include_external=False,
            filter_chain=FilterChain([URLPatternFilter(["*core/*", "*api/*", "*quickstart*"])]),
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=False,
        verbose=False,
    )
    async with AsyncWebCrawler(config=BrowserConfig(text_mode=True, light_mode=True, verbose=False)) as crawler:
        results = await crawler.arun(ROOT_URL, config=config)
    urls = [result.url for result in results if getattr(result, "success", False)]
    return list(dict.fromkeys(urls))


async def process_urls(urls: list[str]) -> list:
    api_schema = {
        "name": "DocHeadings",
        "baseSelector": "main h1, main h2",
        "fields": [{"name": "heading", "type": "text"}],
    }
    configs = [
        CrawlerRunConfig(
            url_matcher=["*api/*"],
            match_mode=MatchMode.OR,
            extraction_strategy=JsonCssExtractionStrategy(api_schema),
            check_robots_txt=True,
            cache_mode=CacheMode.BYPASS,
            verbose=False,
        ),
        CrawlerRunConfig(
            url_matcher=["*core/*", "*quickstart*"],
            match_mode=MatchMode.OR,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=BM25ContentFilter(
                    user_query="browser config extraction markdown crawl result",
                    bm25_threshold=1.0,
                )
            ),
            check_robots_txt=True,
            cache_mode=CacheMode.BYPASS,
            verbose=False,
        ),
        CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.48)
            ),
            check_robots_txt=True,
            cache_mode=CacheMode.BYPASS,
            verbose=False,
        ),
    ]
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0,
        max_session_permit=4,
        rate_limiter=RateLimiter(base_delay=(0.5, 1.0), max_retries=2),
    )
    async with AsyncWebCrawler(config=BrowserConfig(text_mode=True, light_mode=True, verbose=False)) as crawler:
        return await crawler.arun_many(urls, config=configs, dispatcher=dispatcher)


async def main() -> None:
    output_dir, markdown_dir = ensure_output_dir()
    started = time.perf_counter()
    urls = await discover_urls()
    if not urls:
        print("No URLs discovered during the deep-crawl stage.")
        return

    results = await process_urls(urls)
    manifest = []
    success_count = 0
    for result in results:
        markdown = getattr(result.markdown, "fit_markdown", None) or getattr(
            result.markdown, "raw_markdown", str(result.markdown)
        )
        markdown_path = markdown_dir / f"{slugify(result.url)}.md"
        markdown_path.write_text(markdown or "", encoding="utf-8")
        success_count += int(bool(result.success))
        manifest.append(
            {
                "url": result.url,
                "success": result.success,
                "status_code": result.status_code,
                "markdown_file": markdown_path.name,
                "extracted_content_preview": (result.extracted_content or "")[:240],
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"Discovered URLs: {len(urls)}")
    print(f"Processed results: {len(results)}")
    print(f"Successful results: {success_count}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
