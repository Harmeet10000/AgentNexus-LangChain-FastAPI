import asyncio
import json
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v09_schema_power_moves.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, JsonCssExtractionStrategy

# Target URL — the Crawl4AI quickstart docs page used as our extraction subject
URL = "https://docs.crawl4ai.com/core/quickstart/"

################################ Schema Definition ###########################

# JsonCssExtractionStrategy accepts a schema dict that describes *what* to
# extract and *how* to structure the output.  This schema demonstrates three
# advanced features:
#
#   1. baseSelector  — scopes ALL extraction to a single root element.
#                      Using "main" means only content inside <main> is
#                      considered; headers, footers, and nav bars are ignored.
#
#   2. baseFields    — fields extracted ONCE from the baseSelector element
#                      itself (not repeated).  Use these for page-level
#                      metadata that appears exactly once, such as the <h1>
#                      title.
#
#   3. fields        — repeated or structured fields extracted relative to the
#                      baseSelector.  Two sub-types are used here:
#
#       • nested_list — each matched element becomes its own dict inside a
#                       list.  Sub-fields are extracted *relative to each
#                       matched element*, so you get per-item attribute data
#                       (e.g., the anchor href that lives inside each heading).
#                       Use nested_list when you need structured objects, not
#                       just plain text.
#
#       • list        — similar to nested_list but optimised for simpler
#                       cases where you primarily want a single value per
#                       matched element (e.g., the full text of each <pre>
#                       block).  The distinction matters: nested_list builds
#                       richer per-item dicts, while list keeps the output
#                       flat and concise.
SCHEMA = {
    "name": "Docs article outline",
    # Scope the entire extraction to the <main> element so we skip chrome
    # (nav, sidebar, footer) and focus on the article body
    "baseSelector": "main",
    # baseFields are extracted once from the root element — perfect for
    # page-level data like the primary <h1> heading
    "baseFields": [
        {"name": "page_title", "selector": "h1", "type": "text", "transform": "strip", "default": "Untitled"},
    ],
    "fields": [
        {
            # nested_list: each <h2> or <h3> becomes a dict with both its
            # visible text AND the href of its anchor link — two sub-fields
            # per matched element, so we need the richer nested_list type
            "name": "sections",
            "type": "nested_list",
            "selector": "h2, h3",
            "fields": [
                # The visible heading text, whitespace-normalised
                {"name": "heading", "type": "text", "transform": "strip", "default": ""},
                # The fragment href from the anchor tag inside the heading
                # (e.g., "#installation") — extracted as an attribute value
                {"name": "anchor", "selector": "a", "type": "attribute", "attribute": "href", "default": ""},
            ],
        },
        {
            # list: each <pre> block yields a single text snippet — only one
            # sub-field per element, so the simpler list type is appropriate
            "name": "code_blocks",
            "type": "list",
            "selector": "pre",
            "fields": [
                # Full text content of the code block, stripped of leading/
                # trailing whitespace
                {"name": "snippet", "type": "text", "transform": "strip", "default": ""},
            ],
        },
    ],
}

############################ Main Crawl Routine ##############################


async def main() -> None:
    """Crawl the Crawl4AI quickstart page and extract a structured outline.

    Demonstrates the power-move features of JsonCssExtractionStrategy:
    - baseSelector to scope extraction to <main>
    - baseFields for one-shot page-level metadata (page title)
    - nested_list to capture per-heading dicts with both text and anchor href
    - list to collect all code block snippets as a flat array

    Prints the page title, total section count, and total code block count.
    """
    # Wire up the extraction strategy with our schema — verbose=False keeps
    # the console output focused on our own print statements
    strategy = JsonCssExtractionStrategy(SCHEMA, verbose=False)

    # Build the crawler run config:
    # - BYPASS cache to always fetch the live page
    # - Pass the extraction strategy so Crawl4AI runs it automatically after
    #   rendering the page
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, extraction_strategy=strategy, verbose=False)

    # Launch the headless browser session, crawl the page, and let the
    # extraction strategy parse the DOM before returning the result
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=URL, config=config)

    # Guard against network errors, timeouts, or selector mismatches before
    # attempting to parse the extracted payload
    if not result.success:
        print("schema extraction failed:", result.error_message)
        return

    # extracted_content is a JSON string containing a list of records — one
    # record per baseSelector match (here, one record for the single <main>)
    rows = json.loads(result.extracted_content or "[]")
    record = rows[0] if rows else {}

    # Report the page-level title captured via baseFields
    print("title:", record.get("page_title"))
    # Report how many headings were captured by the nested_list "sections" field
    print("section_count:", len(record.get("sections", [])))
    # Report how many code blocks were captured by the list "code_blocks" field
    print("code_block_count:", len(record.get("code_blocks", [])))


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to drive the async
# main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
