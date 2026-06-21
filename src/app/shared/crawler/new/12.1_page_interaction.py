import asyncio
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v13_page_interaction.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

# Define the target URL — a simple, stable page ideal for demonstrating
# JavaScript injection without external dependencies
URL = "https://example.com/"

########################### JavaScript Injection #############################

# JS_CODE is a snippet of JavaScript that Crawl4AI will inject and execute
# inside the page's browser context before capturing content.
#
# insertAdjacentHTML("beforeend", ...) appends the provided HTML string
# immediately before the closing </body> tag, making it the last child of
# <body>. We inject a <div> with the class "js-ready" so that:
#   1. The injected element is detectable in the final HTML output.
#   2. Crawl4AI's wait_for mechanism can use ".js-ready" as a readiness signal.
JS_CODE = """
document.body.insertAdjacentHTML(
  "beforeend",
  "<div class='js-ready'><strong>Injected by js_code</strong></div>"
);
"""

############################ Main Crawl Routine ##############################


async def main() -> None:
    """Crawl a URL with JavaScript injection and verify the injected element.

    This demonstrates how to use js_code and wait_for together in
    CrawlerRunConfig: js_code runs arbitrary JavaScript inside the page, and
    wait_for blocks Crawl4AI from capturing the page until the specified CSS
    selector appears in the DOM — confirming the injection completed before
    content extraction begins.
    """
    # Build the crawler run configuration:
    # - BYPASS cache so we always hit the live page and execute fresh JS
    # - js_code injects our custom HTML element into the page at runtime
    # - wait_for=".js-ready" tells Crawl4AI to pause and poll the DOM until
    #   an element matching that CSS selector exists — acting as a synchronisation
    #   gate to ensure the injection has fully executed before the page is captured
    # - Disable verbose logging to keep console output clean
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        js_code=JS_CODE,
        wait_for=".js-ready",
        verbose=False,
    )

    # Launch an async crawler session using a context manager, which handles
    # browser startup and teardown (Crawl4AI uses a headless browser under the hood)
    async with AsyncWebCrawler() as crawler:
        # Perform the crawl — arun() loads the page, executes js_code, waits
        # for ".js-ready" to appear, then captures the final page state
        result = await crawler.arun(url=URL, config=config)

    # Report whether the overall crawl request succeeded
    print("success:", result.success)

    # Verify the injection actually ran by checking result.html for the "js-ready"
    # class name. If the string is present, the injected <div> made it into the
    # captured HTML, confirming that js_code executed and wait_for resolved correctly.
    print("js_ready_present:", "js-ready" in (result.html or ""))


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
