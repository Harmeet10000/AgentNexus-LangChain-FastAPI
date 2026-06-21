"""Video 5.3 - Fit Markdown Filters
This script demonstrates how to use Crawl4AI's content filters to produce "fit_markdown" — a more concise version of the full "raw_markdown" output that's tailored to a specific query or relevance threshold. It compares two different filter strategies, PruningContent Filter and BM25ContentFilter, by crawling the same page with each and printing out the size of the raw vs fit markdown along with a short preview of the fit output.
Demonstrates:
- DefaultMarkdownGenerator with content filters
- PruningContentFilter and BM25ContentFilter
- raw_markdown vs fit_markdown output
Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

"""

import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from rich import print

# Define the target URL to crawl — the official Crawl4AI documentation site
URL = "https://en.wikipedia.org/wiki/Machine_learning"


########################## Filter Comparison Helper ##########################
async def compare_filter(label: str, generator: DefaultMarkdownGenerator) -> None:
    """Crawl the target URL using the given markdown generator and print a size comparison."""
    # Attach the generator to the run config so Crawl4AI uses the chosen
    # filter strategy when converting the crawled HTML to markdown
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, markdown_generator=generator, verbose=False
    )

    # Launch an async crawler session — the context manager handles headless
    # browser startup and teardown automatically
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=URL, config=config)

    # Guard against network errors or other crawl failures before touching
    # the result payload
    if not result.success:
        print(label, "failed:", result.error_message)
        return

    # Extract the full, unfiltered markdown produced from the crawled page —
    # everything Crawl4AI extracted before any content filter ran.
    markdown = result.markdown
    if hasattr(markdown, "raw_markdown"):
        raw_text = markdown.raw_markdown or ""
    else:
        raw_text = str(markdown or "")

    # Extract the filtered markdown — only the blocks that survived the
    # content filter attached to the generator. Comparing its length to
    # raw_text shows how much the filter removed.
    if hasattr(markdown, "fit_markdown"):
        fit_text = markdown.fit_markdown or ""
    else:
        fit_text = ""
    fit_preview = fit_text[:160].replace("\n", " ").strip()

    # Print a one-line  summary: label, raw size, fit size, and a short preview
    # of the fit output so we can visually verify what was kept
    print(
        label,
        "raw=",
        len(raw_text),
        "fit=",
        len(fit_text),
        "preview=",
        fit_preview,
    )


################################# Main Routine ###############################
async def main() -> None:
    """Build two content-filter generators and compare their output on the same page."""
    # Configure a pruning-based generator — no query required; the filter
    # decides relevance purely from the page's own statistical distribution
    pruning = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.45, threshold_type="dynamic", min_word_threshold=5
        )
    )

    # Configure a BM25-based generator — the query string steers which blocks
    # are considered relevant to the topic we care about
    bm25 = DefaultMarkdownGenerator(
        content_filter=BM25ContentFilter(
            user_query="Types of Machine Learning", bm25_threshold=1.0
        )
    )

    # Run both filters against the same URL so the size and content differences
    # are directly comparable in the printed output
    await compare_filter("pruning", pruning)
    await compare_filter("bm25", bm25)


################################# Entry Point ################################
# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
