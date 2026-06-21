"""Video 20: Multi-URL crawling with intelligent dispatchers.

Demonstrates:
- arun_many() with MemoryAdaptiveDispatcher
- RateLimiter and optional monitor wiring
- URL-specific configs for docs, PDFs, and defaults

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_20_multi_url_dispatchers.py`
"""

import asyncio
import inspect

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerMonitor,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    DisplayMode,
    MatchMode,
    PruningContentFilter,
    RateLimiter,
)
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from crawl4ai.processors.pdf import PDFContentScrapingStrategy

URLS = [
    "https://docs.crawl4ai.com/core/quickstart/",
    "https://docs.crawl4ai.com/core/browser-crawler-config/",
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "https://example.com/",
]


def build_monitor() -> CrawlerMonitor:
    params = inspect.signature(CrawlerMonitor).parameters
    kwargs = {}
    if "display_mode" in params:
        kwargs["display_mode"] = DisplayMode.AGGREGATED
    if "enable_ui" in params:
        kwargs["enable_ui"] = False
    return CrawlerMonitor(**kwargs)


async def main() -> None:
    configs = [
        CrawlerRunConfig(
            url_matcher="*.pdf",
            scraping_strategy=PDFContentScrapingStrategy(),
            verbose=False,
        ),
        CrawlerRunConfig(
            url_matcher=["*docs.crawl4ai.com/core/*", "*docs.crawl4ai.com/api/*"],
            match_mode=MatchMode.OR,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.48)
            ),
            verbose=False,
        ),
        CrawlerRunConfig(verbose=False),
    ]

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0,
        max_session_permit=4,
        rate_limiter=RateLimiter(base_delay=(0.5, 1.0), max_retries=2),
        monitor=build_monitor(),
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(URLS, config=configs, dispatcher=dispatcher)

    print(f"URLs processed: {len(results)}")
    for result in results:
        dispatch = getattr(result, "dispatch_result", None)
        duration = ""
        if dispatch and getattr(dispatch, "start_time", None) and getattr(dispatch, "end_time", None):
            duration = f" duration={dispatch.end_time - dispatch.start_time}"
        print(f"Result: success={result.success} url={result.url}{duration}")


if __name__ == "__main__":
    asyncio.run(main())
