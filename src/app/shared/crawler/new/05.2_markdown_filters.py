"""
5.2 — Markdown Generation Made Easy
-------------------------------------------------
This script walks through 5 progressive demos showing how to turn any
webpage into clean, LLM-ready Markdown using Crawl4AI.

You'll see how to:
- Generate basic Markdown with a single line of code.
- Customize the HTML→Markdown conversion with options.
- Use content filters to prune boilerplate and focus on what's important.

Install:
    crawl4ai-setup
"""

import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter, PruningContentFilter
from rich import print

# The page we'll crawl across all demos. Swap this with any URL you like.
URL = "https://en.wikipedia.org/wiki/Machine_learning"


# ──────────────────────────────────────────────────────────────
# DEMO 1: Basic markdown generation (no filtering)
# ──────────────────────────────────────────────────────────────
async def demo_basic():
    """The simplest possible crawl: fetch the page and return Markdown."""
    print("\n=== DEMO 1: Basic Markdown ===")

    # CrawlerRunConfig tells the crawler HOW to handle this run.
    # Here we only specify the markdown generator — defaults handle the rest.
    config = CrawlerRunConfig(markdown_generator=DefaultMarkdownGenerator())

    # `async with` ensures the browser launches and shuts down cleanly.
    async with AsyncWebCrawler() as crawler:
        # arun() = "async run" — performs a single crawl and returns a CrawlResult
        result = await crawler.arun(URL, config=config)

        # Always check `result.success` before using output (network errors, etc.)
        if result.success:
            # `result.markdown` is a MarkdownGenerationResult object.
            # `.raw_markdown` = unfiltered Markdown of the entire cleaned page.
            print(result.markdown.raw_markdown[:400], "...\n")


# ──────────────────────────────────────────────────────────────
# DEMO 2: Customize markdown options (ignore links, wrap text)
# ──────────────────────────────────────────────────────────────
async def demo_custom_options():
    """Tweak the HTML→Markdown conversion via the `options` dict."""
    print("\n=== DEMO 2: Custom Options ===")

    # Build a generator with custom html2text-style options:
    #   ignore_links  -> strip [text](url) hyperlinks entirely
    #   ignore_images -> drop ![alt](src) image tags
    #   body_width    -> hard-wrap text lines at N characters (0 = no wrap)
    md_generator = DefaultMarkdownGenerator(
        options={"ignore_links": True, "ignore_images": True, "body_width": 80}
    )
    config = CrawlerRunConfig(markdown_generator=md_generator)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL, config=config)
        if result.success:
            # Notice how the output is cleaner — no link clutter or image refs.
            print(result.markdown.raw_markdown[:400], "...\n")


# ──────────────────────────────────────────────────────────────
# DEMO 3: PruningContentFilter — strip boilerplate noise
# ──────────────────────────────────────────────────────────────
async def demo_pruning_filter():
    """Use heuristic pruning to remove low-value content blocks (nav, ads)."""
    print("\n=== DEMO 3: Pruning Filter (fit_markdown) ===")

    md_generator = DefaultMarkdownGenerator(
        # PruningContentFilter scores every block (text density, link density,
        # tag patterns) and drops anything below the threshold.
        content_filter=PruningContentFilter(
            threshold=0.5,  # keep blocks scoring >= 0.5
            threshold_type="fixed",  # "fixed" = absolute cutoff; "dynamic" = adaptive
            min_word_threshold=50,  # discard blocks with < 50 words
        ),
        options={"ignore_links": True},
    )
    config = CrawlerRunConfig(markdown_generator=md_generator)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL, config=config)
        if result.success:
            # When a content_filter is set, you get BOTH outputs:
            #   raw_markdown -> the full page in Markdown
            #   fit_markdown -> the pruned, focused Markdown
            print(f"Raw length:  {len(result.markdown.raw_markdown)}")
            print(f"Fit length:  {len(result.markdown.fit_markdown)}")
            print("--- fit_markdown preview ---")
            print(result.markdown.fit_markdown[:400], "...\n")


# ──────────────────────────────────────────────────────────────
# DEMO 4: BM25ContentFilter — query-focused extraction
# ──────────────────────────────────────────────────────────────
async def demo_bm25_filter():
    """Keep only the page sections that are relevant to a search query."""
    print("\n=== DEMO 4: BM25 Filter (query-driven) ===")

    md_generator = DefaultMarkdownGenerator(
        # BM25 is a classic information-retrieval ranking algorithm.
        # It scores each block against `user_query` and keeps the top matches.
        content_filter=BM25ContentFilter(
            user_query="neural networks deep learning",  # what we care about
            bm25_threshold=1,  # higher = stricter (fewer blocks kept)
            language="english",  # used for stemming ("running" -> "run")
        ),
        options={"ignore_links": True},
    )
    config = CrawlerRunConfig(markdown_generator=md_generator)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL, config=config)
        if result.success:
            # fit_markdown will contain ONLY the query-relevant chunks.
            print(result.markdown.fit_markdown[:400], "...\n")


# ──────────────────────────────────────────────────────────────
# DEMO 5: Inspect the full MarkdownGenerationResult object
# ──────────────────────────────────────────────────────────────
async def demo_result_object():
    """Explore every field returned in `result.markdown`."""
    print("\n=== DEMO 5: MarkdownGenerationResult Fields ===")

    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.9)
        )
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL, config=config)

        # `result.markdown` exposes 4 useful string fields:
        md = result.markdown
        #   raw_markdown            -> direct HTML→Markdown conversion
        #   markdown_with_citations -> links converted to [text][1] footnotes
        #   references_markdown     -> the [1]: https://... footnote section
        #   fit_markdown            -> the filtered/pruned version
        print(f"raw_markdown            : {len(md.raw_markdown)} chars")
        print(f"markdown_with_citations : {len(md.markdown_with_citations)} chars")
        print(f"references_markdown     : {len(md.references_markdown)} chars")
        print(f"fit_markdown            : {len(md.fit_markdown)} chars")


# ──────────────────────────────────────────────────────────────
# Entry point — run all demos sequentially
# ──────────────────────────────────────────────────────────────
async def main():
    # Each demo opens its own browser; running sequentially keeps output readable.
    # In production, reuse a single AsyncWebCrawler for many URLs.
    await demo_basic()
    await demo_custom_options()
    await demo_pruning_filter()
    await demo_bm25_filter()
    await demo_result_object()


if __name__ == "__main__":
    # asyncio.run() boots the event loop and runs main() to completion.
    asyncio.run(main())
