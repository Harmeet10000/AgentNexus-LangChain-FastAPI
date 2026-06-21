import asyncio
import json
import sys
from pathlib import Path
from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
)

# Define the target URL to crawl — the official Crawl4AI documentation site
URL = "https://docs.crawl4ai.com/"

############################ Extraction Schema ###############################

# JsonCssExtractionStrategy uses a declarative schema to map CSS selectors
# directly onto structured JSON output — no custom parsing code required.
#
# Schema keys:
#   name          - human-readable label for the schema (used in logging)
#   baseSelector  - CSS selector that identifies each repeating item on the
#                   page; one output record is produced per matched element
#   fields        - list of field descriptors that define what data to pull
#                   out of each matched element
SCHEMA = {
    "name": "Docs navigation links",
    # Match every <a> element inside any <nav> — each anchor becomes one row
    # in the output list
    "baseSelector": "nav a",
    "fields": [
        # "text" field: grab the visible inner text of the element.
        # type="text"        — extract the element's text content
        # transform="strip"  — trim leading/trailing whitespace from the value
        {"name": "text", "type": "text", "transform": "strip"},
        # "href" field: pull a specific HTML attribute value from the element.
        # selector="a"       — look for an <a> tag within (or at) the base element
        # type="attribute"   — extract an attribute rather than text content
        # attribute="href"   — the specific attribute to read
        # default=""         — fall back to an empty string if the attribute is absent
        {
            "name": "href",
            "selector": "a",
            "type": "attribute",
            "attribute": "href",
            "default": "",
        },
    ],
}


############################ Main Crawl Routine ##############################


async def main() -> None:
    """Crawl a single URL and extract structured data using CSS selectors.

    This demonstrates JsonCssExtractionStrategy: define a schema that maps
    CSS selectors to field names and Crawl4AI will return a JSON array of
    records — one per element matched by baseSelector — stored in
    result.extracted_content.
    """
    # Attach the extraction strategy to a CrawlerRunConfig so it runs
    # automatically after the page is fetched and rendered
    strategy = JsonCssExtractionStrategy(SCHEMA, verbose=False)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, extraction_strategy=strategy, verbose=False
    )

    # Launch an async crawler session using a context manager, which handles
    # browser startup and teardown (Crawl4AI uses a headless browser under the hood)
    async with AsyncWebCrawler() as crawler:
        # arun() fetches and renders the page, then applies the extraction
        # strategy — the structured output lands in result.extracted_content
        result = await crawler.arun(url=URL, config=config)

    # Guard against crawl failures (e.g., network errors, timeouts) before
    # attempting to access the result payload
    if not result.success:
        print("extraction failed:", result.error_message)
        return

    # extracted_content is a raw JSON string (or None if nothing was extracted).
    # Parse it into a Python list; fall back to an empty list if it is absent.
    rows = json.loads(result.extracted_content or "[]")

    # Print a quick summary so we can verify the extraction worked as expected
    print("items:", len(rows))
    print("first:", rows[0] if rows else None)


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
