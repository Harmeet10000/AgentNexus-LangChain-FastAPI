import asyncio
import json
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v10_regex_extraction.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, RegexExtractionStrategy

# Define the target URL to crawl — the official Python 3 documentation site
URL = "https://docs.python.org/3/"

########################## Regex Extraction Routine ##########################


async def main() -> None:
    """Demonstrate RegexExtractionStrategy with built-in and custom patterns.

    Two strategies are built and run against the same page in sequence:

    1. built_in — combines Crawl4AI's pre-defined Url and Email patterns using
       bitwise OR, so a single pass collects both types of matches.
    2. custom — supplies a hand-written regex under a named key so results
       are labelled in the output JSON.

    Both strategies operate on the raw HTML source (input_format="html"),
    meaning the regex runs before any markdown conversion or tag stripping.
    """
    # --- Built-in pattern strategy ---
    # RegexExtractionStrategy ships with pre-compiled patterns for common data
    # types (Url, Email, PhoneNumber, etc.).  Combining them with the bitwise
    # OR operator (|) tells the strategy to run all selected patterns in a
    # single scan and merge the results into one list.
    built_in = RegexExtractionStrategy(
        pattern=RegexExtractionStrategy.Url | RegexExtractionStrategy.Email,
        # Scan the raw HTML rather than the rendered markdown, so href values
        # and mailto links hidden inside attributes are also captured.
        input_format="html",
    )

    # --- Custom pattern strategy ---
    # The `custom` argument accepts a dict that maps a label (used as the
    # "type" key in each result object) to a regex string.  The pattern below
    # matches version strings like "Python 3.12" or "Python 3.12.4":
    #   Python\s+  — the literal word "Python" followed by one or more spaces
    #   3\.        — the major version digit "3" and a literal dot
    #   \d+        — one or more digits for the minor version
    #   (?:\.\d+)? — an optional non-capturing group for the patch version
    custom = RegexExtractionStrategy(
        custom={"python_version": r"Python\s+3\.\d+(?:\.\d+)?"},
        # Again, scan raw HTML so version strings in <title>, meta tags, and
        # other non-visible elements are included.
        input_format="html",
    )

    # Launch a single crawler session and perform both extractions back-to-back.
    # Using one AsyncWebCrawler context reuses the underlying browser instance,
    # which avoids the overhead of starting a new headless browser for each run.
    async with AsyncWebCrawler() as crawler:
        # First pass: collect URLs and email addresses from the page HTML
        built_in_result = await crawler.arun(
            url=URL,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, extraction_strategy=built_in, verbose=False),
        )
        # Second pass: extract Python version strings using the custom pattern
        custom_result = await crawler.arun(
            url=URL,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, extraction_strategy=custom, verbose=False),
        )

    ############################# Results Output #############################

    # extracted_content is a JSON string when an extraction strategy is used.
    # Fall back to an empty list literal so json.loads() never receives None.
    if built_in_result.success:
        matches = json.loads(built_in_result.extracted_content or "[]")
        print("built-in matches:", len(matches), "sample:", matches[:3])

    if custom_result.success:
        matches = json.loads(custom_result.extracted_content or "[]")
        print("custom matches:", len(matches), "sample:", matches[:3])


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
