import asyncio
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v02_config_objects.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

# Define the target URL to crawl — the official Crawl4AI documentation site
URL = "https://docs.crawl4ai.com/"

############################### Helper Function ##############################


async def run_once(crawler: AsyncWebCrawler, label: str, config: CrawlerRunConfig) -> None:
    """Crawl the target URL once with a given config and print a summary line.

    This helper keeps main() tidy by encapsulating the crawl + print pattern.
    The label argument identifies which configuration profile was used so the
    two runs are easy to distinguish in terminal output.
    """
    # Execute the crawl using the pre-opened crawler session and the supplied config
    result = await crawler.arun(url=URL, config=config)

    # Print a compact summary: the config label, whether the crawl succeeded,
    # and the size of the raw HTML payload as a quick sanity check
    print(label, "success=", result.success, "html_chars=", len(result.html or ""))
    print(result.markdown[:200]) 


################################# Main Routine ###############################


async def main() -> None:
    """Demonstrate BrowserConfig and CrawlerRunConfig as first-class objects.

    Crawl4AI separates browser-level settings (BrowserConfig) from per-crawl
    settings (CrawlerRunConfig). This script shows both config objects being
    constructed explicitly and then reused across multiple crawl calls — a
    pattern that scales well when you need different behaviours for different
    pages within the same browser session.
    """
    # --- Browser-level configuration ---
    # BrowserConfig controls the underlying Playwright browser instance that
    # Crawl4AI manages for you. These settings apply for the entire session.
    browser_config = BrowserConfig(
        browser_type="chromium",   # Use Chromium (alternatives: "firefox", "webkit")
        headless=True,             # Run without a visible window — ideal for scripts
        viewport_width=1280,       # Emulate a typical desktop screen width
        viewport_height=720,       # Emulate a typical desktop screen height
        verbose=False,             # Suppress low-level browser log noise
    )

    # --- Per-crawl configuration: fast profile ---
    # A minimal config that simply bypasses the cache and stays silent.
    # Use this when you want a quick, clean fetch with no extra waiting.
    fast_run = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=False)

    # --- Per-crawl configuration: debug profile ---
    # A richer config that trades speed for thoroughness and visibility.
    debug_run = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,        # Always fetch live — skip any cached response
        wait_until="networkidle",           # Wait until network traffic settles before
                                            # capturing the page (catches late-loading JS)
        exclude_external_links=True,        # Strip outbound links from the result to keep
                                            # the output focused on the page's own content
        verbose=True,                       # Print detailed progress so you can trace
                                            # exactly what the crawler is doing
    )

    # Open a single browser session and reuse it for both crawl profiles.
    # AsyncWebCrawler accepts a BrowserConfig directly — no need to pass it
    # again to individual arun() calls.
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Run the fast profile first — quick baseline fetch
        await run_once(crawler, "fast", fast_run)

        # Run the debug profile second — slower but more thorough, with verbose output
        await run_once(crawler, "debug", debug_run)


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
