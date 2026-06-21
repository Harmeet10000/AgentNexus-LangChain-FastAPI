"""
Title: Cache Modes Comparison

Description:
This script demonstrates the three primary cache modes in Crawl4AI by timing how long it takes to
crawl the same URL under each mode. The modes are:
- ENABLED: Uses the cache if available, otherwise fetches live and updates the cache.
- READ_ONLY: Only reads from the cache; never writes back. Fast if the cache is populated, but may return stale data.
- BYPASS: Ignores the cache entirely; always fetches live from the network and does not store results. Useful for development when you need fresh content every time.
"""

import asyncio
import time
from rich import print
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

# Define the target URL to crawl — the official Crawl4AI documentation site
URL = "https://docs.crawl4ai.com/marketplace/"


###################### Timed Crawl Helper ######################
async def timed_crawl(crawler: AsyncWebCrawler, label: str, mode: CacheMode) -> None:
    """Crawl the target URL under a given CacheMode and print how long it took.

    This helper isolates each cache-mode experiment so the main() function can
    run them back-to-back with minimal boilerplate. Timing is captured with a
    high-resolution counter to make the speed difference between modes obvious.

    Args:
        crawler: An already-open AsyncWebCrawler session to reuse across calls.
        label:   A short string printed alongside the result so each line is
                 easy to identify in the terminal output.
        mode:    The CacheMode to apply for this crawl (see main() for details).
    """
    # Record the wall-clock start time before issuing the crawl request.
    # perf_counter() provides sub-millisecond resolution and is not affected
    # by system clock adjustments, making it ideal for elapsed-time measurement.
    started = time.perf_counter()

    # Execute the crawl with the requested cache mode. verbose=False suppresses
    # Crawl4AI's internal progress logs so only our own print() output is shown.
    result = await crawler.arun(
        url=URL, config=CrawlerRunConfig(cache_mode=mode, verbose=False)
    )

    # Compute elapsed time immediately after the crawl completes so we capture
    # only the crawl latency, not any later processing overhead.
    elapsed = time.perf_counter() - started

    # Print a single summary line per crawl: the label, whether it succeeded,
    # and how many seconds it took (two decimal places for readability).
    print(label, "success=", result.success, "seconds=", f"{elapsed:.2f}")


############################ Main Demo ############################
async def main() -> None:
    """Compare Crawl4AI's three primary cache modes by timing the same URL.

    Crawl4AI maintains a local content cache so repeated visits to the same
    page can skip the network round-trip.  There are three modes worth knowing:

    - ENABLED   — read from cache if a stored copy exists; write to cache if not.
                  The first call here will fetch live and populate the cache;
                  subsequent ENABLED calls would be nearly instant.

    - READ_ONLY — only read from cache; never write back.  Useful when you want
                  cached data without risking a stale write, or when you are
                  doing a dry-run over previously cached content.

    - BYPASS    — ignore the cache entirely; always fetch live from the network
                  and do not store the result.  Handy during development when
                  you need fresh content every time regardless of cache state.

    Running all three in sequence lets you observe the latency difference
    between a live fetch and a cache hit in a single terminal session.
    """
    # Open a single browser session and reuse it across all three crawls.
    # Sharing the session avoids the overhead of launching a new headless
    # browser for each call, keeping the timing comparison fair.
    async with AsyncWebCrawler() as crawler:
        # First crawl: ENABLED mode — this will hit the network because the
        # cache is cold (or stale), then store the result for future reads.
        await timed_crawl(crawler, "enabled-first", CacheMode.ENABLED)

        # Second crawl: READ_ONLY mode — reads the entry written above without
        # updating the cache.  Expect a much shorter elapsed time compared to
        # the live fetch above, demonstrating the cache speed-up.
        await timed_crawl(crawler, "read-only", CacheMode.READ_ONLY)

        # Third crawl: BYPASS mode — skips the cache entirely and goes back to
        # the live network.  Elapsed time should be similar to the first crawl,
        # confirming that BYPASS always pays the full network cost.
        await timed_crawl(crawler, "bypass", CacheMode.BYPASS)


############################ Entry Point ########################
if __name__ == "__main__":
    asyncio.run(main())
