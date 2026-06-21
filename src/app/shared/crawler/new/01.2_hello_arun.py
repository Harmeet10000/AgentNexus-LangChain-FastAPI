"""Video 01: Your hello crawl to Crawl4AI docs.

Demonstrates:
- AsyncWebCrawler and arun()
- CrawlResult success, markdown, html, cleaned_html, status_code
- Minimal error handling

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`
"""

import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from rich import print

# Define the target URL to crawl — the official Crawl4AI documentation site
URL = "https://docs.crawl4ai.com/"


############################ Main Crawl Routine ##############################
async def main() -> None:
    """Crawl a single URL using Crawl4AI and print a truncated markdown preview.

    This demonstrates the minimal setup needed to fetch a page with
    AsyncWebCrawler: create a run config, open a crawler context, and
    inspect the result object.
    """
    # Build the crawler run configuration:
    # - BYPASS cache so we always hit the live page instead of returning stale data
    # - Disable verbose logging to keep the console output clean
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=True)

    # Launch an async crawler session using a context
    # manager, which handles browser startup and teardown
    async with AsyncWebCrawler() as crawler:
        # Perform the actual crawl — arun() fetches the page, renders any
        # JavaScript, and converts the visible content into markdown
        result = await crawler.arun(url=URL, config=config)

    # Guard against crawl failures (e.g., network errors, timeouts) before
    # attempting to access the result payload
    if not result.success:
        print("crawl failed:", result.error_message)
        return

    # Display the final resolved URL (may differ from the input if redirects occurred)
    print("result:", result)
    print("url:", result.url)
    # print result object for reference without links
    print("result:", result.metadata)
    # Extract the raw markdown from the result and show only the first 500 chars
    print("Markdown:\n", result.markdown[:500])
    print("HTML: \n", result.cleaned_html[:500])
    # print html and cleaned_html lengths for reference
    print("html length:", len(result.html or ""))
    print("cleaned_html length:", len(result.cleaned_html or ""))
    internal = result.links.get("internal", [])
    external = result.links.get("external", [])
    print(f"Internal links: {len(internal)}")
    print(f"External links: {len(external)}")


################################# Entry Point ################################
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
