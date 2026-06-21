import asyncio
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v17_arun_many_dispatchers.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerMonitor,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)

############################# Target URLs ####################################

# A small list of Crawl4AI documentation pages used to demonstrate batch and
# streaming modes — four URLs is enough to show concurrency without being noisy
URLS = [
    "https://docs.crawl4ai.com/",
    "https://docs.crawl4ai.com/core/quickstart/",
    "https://docs.crawl4ai.com/core/browser-crawler-config/",
    "https://docs.crawl4ai.com/core/markdown-generation/",
]

########################### Dispatcher Factory ###############################


def build_dispatcher(total: int) -> MemoryAdaptiveDispatcher:
    """Build a memory-aware dispatcher that throttles concurrent crawls.

    MemoryAdaptiveDispatcher monitors system RAM usage and pauses new tasks
    when memory climbs above the configured threshold, preventing out-of-memory
    crashes on large batches.  Each call to this factory creates a fresh
    dispatcher with its own monitor so that batch and stream runs are tracked
    independently.

    Args:
        total: The total number of URLs in the batch, passed to CrawlerMonitor
               so it can report meaningful progress percentages.

    Returns:
        A configured MemoryAdaptiveDispatcher ready to be handed to arun_many().
    """
    return MemoryAdaptiveDispatcher(
        # Pause scheduling new tasks when system memory usage exceeds 80 %
        memory_threshold_percent=80.0,
        # Poll memory usage every second while the batch is running
        check_interval=1.0,
        # Allow at most 4 browser sessions to run concurrently at any time
        max_session_permit=4,
        # RateLimiter inserts a random pause between 0.5 and 1.0 seconds before
        # each request, backing off up to 5 seconds on failures and retrying up
        # to twice — this prevents hammering the target server
        rate_limiter=RateLimiter(base_delay=(0.5, 1.0), max_delay=5.0, max_retries=2),
        # CrawlerMonitor tracks how many URLs have finished vs. the total so
        # progress can be logged; enable_ui=False keeps output plain-text
        monitor=CrawlerMonitor(urls_total=total, enable_ui=False),
    )


############################ Main Crawl Routine ##############################


async def main() -> None:
    """Demonstrate arun_many() in both batch and streaming modes.

    arun_many() crawls a list of URLs concurrently using a dispatcher to manage
    resource usage.  This function shows two complementary patterns:

    - **Batch mode** (stream=False): arun_many() waits for every URL to finish
      before returning the full list of results.  Use this when downstream logic
      needs all pages before it can proceed.

    - **Streaming mode** (stream=True): arun_many() returns an async iterator
      that yields each CrawlResult as soon as its URL completes.  Use this when
      you want to process or display results incrementally without waiting for
      the slowest URL.
    """
    # BrowserConfig controls the underlying browser process shared across all
    # concurrent sessions — headless keeps the UI hidden, verbose=False suppresses
    # low-level browser logs
    browser_config = BrowserConfig(headless=True, verbose=False)

    # Batch run config: stream=False means arun_many() collects all results
    # internally and returns them together as a list once every URL is done
    batch_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False, verbose=False)

    # Stream run config: stream=True means arun_many() returns an async iterator
    # so each result can be consumed as soon as that individual crawl finishes
    stream_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=True, verbose=False)

    # Launch a single browser context that is reused across all concurrent
    # crawl sessions — this is more efficient than opening a new browser per URL
    async with AsyncWebCrawler(config=browser_config) as crawler:

        # --- Batch mode ---
        # arun_many() fans out across all URLS concurrently, respecting the
        # dispatcher's memory and rate limits, then returns the complete list
        batch_results = await crawler.arun_many(URLS, config=batch_config, dispatcher=build_dispatcher(len(URLS)))

        # Count how many pages were fetched successfully and report the ratio
        print("batch_ok:", sum(1 for result in batch_results if result.success), "of", len(batch_results))

        # --- Streaming mode ---
        # Using only the first two URLs keeps this demo output concise; in
        # production you would pass the full list just as in batch mode
        stream_iter = await crawler.arun_many(URLS[:2], config=stream_config, dispatcher=build_dispatcher(2))

        # async for pulls results from the iterator one at a time as they
        # arrive — each iteration yields a CrawlResult for a completed URL
        async for result in stream_iter:
            print("stream:", result.success, result.url)


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
