import asyncio
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v15_c4a_script_dsl.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

# Define the target URL to crawl
URL = "https://example.com/"

############################ C4A-Script DSL ##################################

# C4A-Script is Crawl4AI's high-level domain-specific language (DSL) for
# browser automation. It provides a readable, instruction-based alternative to
# raw JavaScript: instead of writing inline JS for every interaction, you
# compose a sequence of named commands that the crawler executes in order.
#
# Each line in the script is one instruction. The supported commands include:
#
#   WAIT <seconds>               — pause execution for a fixed number of seconds
#   WAIT `<css-selector>` <t>    — wait until the CSS selector appears in the DOM,
#                                  with an optional timeout in seconds
#   IF (EXISTS `<selector>`) THEN EVAL `<js>` — conditionally evaluate a JavaScript
#                                  expression only when the given element is present
#   EVAL `<js>`                  — unconditionally evaluate a JavaScript expression
#   CLICK `<selector>`           — click the matched element
#   TYPE `<selector>` `<text>`   — type text into an input field
#
# Using c4a_script= (DSL) vs js_code= (raw JavaScript):
#   - js_code      injects arbitrary JavaScript directly into the page; powerful
#                  but verbose and harder to read at a glance.
#   - c4a_script   wraps common browser-automation patterns in terse, named
#                  commands, making scripts easier to write, read, and debug.
#
# This script demonstrates a simple DSL sequence:
#   1. WAIT 1                  — pause 1 second to let the page settle after load
#   2. IF (EXISTS `body`) THEN EVAL `...`
#                              — if <body> is present, inject a sentinel <div>
#                                into the DOM so we can confirm the script ran
#   3. WAIT `.dsl-ready` 5     — wait up to 5 seconds for that sentinel element
#                                to appear, confirming the EVAL succeeded
SCRIPT = """
WAIT 1
IF (EXISTS `body`) THEN EVAL `document.body.insertAdjacentHTML('beforeend', '<div class="dsl-ready">C4A-Script worked</div>')`
WAIT `.dsl-ready` 5
"""

############################ Main Crawl Routine ##############################


async def main() -> None:
    """Crawl a URL using a C4A-Script DSL program and verify it executed.

    This demonstrates how to pass a c4a_script to CrawlerRunConfig to drive
    browser automation with Crawl4AI's high-level DSL. After crawling, the
    result HTML is inspected to confirm the sentinel element injected by the
    DSL was present, proving the script ran successfully.
    """
    # Build the crawler run configuration:
    # - BYPASS cache so we always hit the live page
    # - c4a_script runs the DSL program above after the page loads
    # - wait_for blocks until the sentinel CSS selector is present in the DOM,
    #   acting as a final synchronisation point before the crawler captures HTML
    # - Disable verbose logging to keep console output clean
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        c4a_script=SCRIPT,
        wait_for=".dsl-ready",
        verbose=False,
    )

    # Launch an async crawler session using a context manager, which handles
    # browser startup and teardown
    async with AsyncWebCrawler() as crawler:
        # Perform the crawl — the DSL script runs automatically as part of the
        # page interaction phase, before the final HTML snapshot is taken
        result = await crawler.arun(url=URL, config=config)

    # Report whether the overall crawl request succeeded
    print("success:", result.success)

    # Confirm the sentinel element injected by the DSL is present in the
    # captured HTML — if True, the EVAL command ran and the DOM was mutated
    print("dsl_ready_present:", "dsl-ready" in (result.html or ""))


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
