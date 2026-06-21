"""Video 02.2: BrowserConfig controls your browser session.

Demonstrates:
- BrowserConfig browser type, viewport, user agent, headless mode
- Using clone() for debug variants
- text_mode/light_mode for lighter crawls

Prerequisites:
- `pip install crawl4ai playwright`
-  `crawl4ai-setup `
"""

import asyncio
import time
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from rich import print

########################### Constants & Defaults ##########################
URL = "https://docs.crawl4ai.com/core/quickstart/"
PRIMARY_BROWSER = "chromium"
LARGE_VIEWPORT = (1440, 900)
MOBILE_VIEWPORT = (430, 932)


############################# Helper Functions #############################
async def crawl_once(label: str, config: BrowserConfig) -> None:
    """Crawl a single URL and print timing and result summary."""
    started = time.perf_counter()
    # Define a simple config with the provided browser settings and some content filtering
    crawler_config = CrawlerRunConfig(excluded_tags=["form", "header", "sidebar"])
    # Open a browser instance, crawl, then auto-close via async context manager
    async with AsyncWebCrawler(config=config) as crawler:
        result = await crawler.arun(URL, config=crawler_config)
    elapsed = time.perf_counter() - started
    print(
        f"'\n\n'{'=' * 40}\nCrawl Result for {label}\n{'=' * 40}"
        f"\n{label}:\nsuccess={result.success}"
        f"\nbrowser={config.browser_type} "
        f"\nviewport={config.viewport_width}x{config.viewport_height} "
        f"\nchars={len(result.markdown)} "
        f"\ntime={elapsed:.2f}s\n"
    )
    # print(result.markdown[:1000] + "...\n")  # Print a snippet of the result


############################### Main Workflow ###############################
async def main() -> None:
    """Configure multiple browser profiles and run crawls to compare results."""
    # Set up the base desktop browser config with randomized user agent
    base_browser = BrowserConfig(
        browser_type=PRIMARY_BROWSER,
        headless=True,  # Run headless mode for faster performance avoid opening windows
        viewport_width=LARGE_VIEWPORT[0],
        viewport_height=LARGE_VIEWPORT[1],
        # User agent is what websites identify your browser type/version
        # e.g "AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36"
        # Rotate to avoid fingerprinting detection, get more realistic results
        user_agent_mode="random",
        verbose=True,
    )
    # Derive a mobile variant by cloning and overriding only the viewport
    mobile_browser = base_browser.clone(
        viewport_width=MOBILE_VIEWPORT[0],
        viewport_height=MOBILE_VIEWPORT[1],
    )
    # Enable text_mode and light_mode to skip images/JS for faster crawls
    text_browser = base_browser.clone(text_mode=True, light_mode=True)
    # Create a visible, verbose config useful for interactive debugging
    debug_browser = base_browser.clone(headless=False, verbose=True)

    print("Running headless large viewport crawl...")
    # Attempt crawls with the primary browser, falling back to chromium on failure

    await crawl_once("desktop", base_browser)
    await crawl_once("mobile", mobile_browser)
    await crawl_once("text-mode", text_browser)
    # await crawl_once("debug", debug_browser)

    print(
        "Debug config ready: "
        f"browser={debug_browser.browser_type}, headless={debug_browser.headless}"
    )


if __name__ == "__main__":
    asyncio.run(main())
