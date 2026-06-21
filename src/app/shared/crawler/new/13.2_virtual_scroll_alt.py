import asyncio
import re
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v16_virtual_scroll.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, VirtualScrollConfig

# Define the target URL to crawl — the official Crawl4AI documentation site
URL = "https://docs.crawl4ai.com/"

############################### Link Counter #################################


def count_links(html: str) -> int:
    """Count the number of anchor tags in an HTML string.

    Uses a simple regex to find all opening <a> tags (case-insensitive).
    This gives a quick proxy for how much content was rendered on the page —
    pages with lazy-loaded navigation or link lists will show more links after
    virtual scrolling has been applied.
    """
    # re.findall returns a list of all non-overlapping matches; its length is
    # the total anchor-tag count. The `or ""` guard handles a None input safely.
    return len(re.findall(r"<a\\b", html or "", flags=re.IGNORECASE))


############################ Main Crawl Routine ##############################


async def main() -> None:
    """Compare link counts from a plain crawl vs. a virtual-scroll crawl.

    Many modern sites use lazy loading — content is only injected into the DOM
    after the user scrolls down. VirtualScrollConfig tells Crawl4AI to simulate
    that scrolling inside the headless browser before capturing the page HTML,
    which reveals content that a single-shot crawl would miss.

    By counting anchor tags in both results we can see how much additional
    content the virtual-scroll pass surfaces.
    """
    # Build the baseline config: no caching, no scrolling, quiet output.
    # This represents what a conventional crawl would capture in one shot.
    base_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=False)

    # Build the virtual-scroll config. VirtualScrollConfig instructs the browser
    # to programmatically scroll the page in steps, pausing between each one so
    # that lazy-loaded content has time to render before the HTML is captured.
    scroll_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        virtual_scroll_config=VirtualScrollConfig(
            # The CSS selector for the element that should be scrolled.
            # "body" targets the main document viewport, which works for most pages.
            container_selector="body",
            # How many scroll steps to perform before capturing the final HTML.
            # More steps expose content further down the page.
            scroll_count=3,
            # How far to scroll on each step. "page_height" advances by one full
            # viewport height, mimicking a user pressing the Page Down key.
            scroll_by="page_height",
            # Seconds to wait after each scroll before the next step.
            # Allows JavaScript-driven content (e.g., infinite-scroll loaders)
            # to finish rendering before the crawler advances.
            wait_after_scroll=0.4,
        ),
        verbose=False,
    )

    # Launch a single headless browser session and run both crawls back-to-back.
    # Reusing the same AsyncWebCrawler instance avoids the overhead of spinning
    # up a second browser process.
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        # Baseline crawl — captures the page exactly as it loads with no scrolling
        base = await crawler.arun(url=URL, config=base_config)
        # Virtual-scroll crawl — scrolls the page before capturing HTML
        scrolled = await crawler.arun(url=URL, config=scroll_config)

    # Print link counts side by side. A higher count in the scrolled result
    # confirms that virtual scrolling surfaced additional lazy-loaded content
    # that the baseline crawl never saw.
    print("base_links:", count_links(base.html or ""))
    print("virtual_scroll_links:", count_links(scrolled.html or ""))


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
