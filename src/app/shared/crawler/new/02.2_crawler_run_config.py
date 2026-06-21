"""Video 02.1: Tuning each crawl with CrawlerRunConfig.

Demonstrates:
- word_count_threshold and excluded_tags for content filtering
- External/social link filtering
- Cache modes (ENABLED vs BYPASS) and config cloning
- wait_for, page_timeout, and remove_overlay_elements
- Accessing MarkdownGenerationResult attributes

Prerequisites:
    pip install crawl4ai playwright rich
    playwright install

Run:
    python crawl4ai_101/video_03_crawler_run_config.py
"""

import asyncio
import time
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from rich import print

####################### Constants & Setup #####################

URL = "https://docs.crawl4ai.com/"


################### Helper Functions ########################


def summarize(label: str, result, elapsed: float) -> None:
    """Print a compact summary of a CrawlResult."""

    internal = result.links.get("internal", [])
    external = result.links.get("external", [])
    md = result.markdown  # MarkdownGenerationResult object
    raw_len = len(md.raw_markdown) if md else 0

    print(f"\n[bold cyan]── {label} ──[/bold cyan]")
    print(f"  URL          : {result.url}")
    print(f"  Status       : {result.status_code}")
    print(f"  Elapsed      : {elapsed:.4f}s")
    print(f"  Markdown len : {raw_len:,} chars")
    print(f"  Internal links: {len(internal)}")
    print(f"  External links: {len(external)}")


##################### Main Crawl Logic ##########################


async def main() -> None:
    """Run two crawls (bypass then cached) and compare results."""

    # ── Base config: content filtering + caching ──
    # Define a reusable configuration with filtering, interaction, and cache settings
    base_config = CrawlerRunConfig(
        # Content filtering
        word_count_threshold=30,  # Skip blocks < 30 words long like navbars
        excluded_tags=["nav", "footer", "header"],  # Strip boilerplate sections
        # Link filtering
        exclude_external_links=False,
        exclude_social_media_links=False,
        # Page interaction
        # remove_overlay_elements=True,  # Dismiss modals/popups
        wait_for="css:main",  # Wait for <main> in DOM
        page_timeout=30_000,  # 30s load timeout
        # Caching & verbosity
        cache_mode=CacheMode.ENABLED,
        verbose=False,
    )

    # Clone the base config and modify only the cache mode for bypassing
    bypass_config = base_config.clone(cache_mode=CacheMode.WRITE_ONLY)

    async with AsyncWebCrawler() as crawler:
        # ── First crawl: bypass cache (populates it) ──
        # Time the fresh fetch so we can compare against the cached version
        t0 = time.perf_counter()
        result_bypass = await crawler.arun(URL, config=bypass_config)
        elapsed_bypass = time.perf_counter() - t0
        summarize("BYPASS (fresh fetch) Write Only", result_bypass, elapsed_bypass)

        # ── Second crawl: read from cache ──
        # Re-crawl the same URL using the default ENABLED cache mode
        t0 = time.perf_counter()
        result_cached = await crawler.arun(URL, config=base_config)
        elapsed_cached = time.perf_counter() - t0
        summarize("ENABLED (from cache)", result_cached, elapsed_cached)

    ################### Compare Results ######################

    # Calculate and display the performance gain from caching
    if result_bypass.success and result_cached.success:
        speedup = elapsed_bypass / elapsed_cached if elapsed_cached else float("inf")
        print(f"\n[bold green]⚡ Cache speedup: {speedup:.1f}×[/bold green]")

        # Show a markdown preview from the bypass result
        raw = result_bypass.markdown.raw_markdown
        preview = raw[:300].strip()
        print(f"\n[bold]Markdown preview (first 300 chars):[/bold]\n{preview}…")


if __name__ == "__main__":
    asyncio.run(main())
