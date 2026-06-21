import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from rich import print

# Define the target URL to crawl — the official Crawl4AI documentation site
URL = "https://docs.crawl4ai.com/"

########################## Per-Source Crawl Routine ##########################


async def crawl_with_source(source: str) -> None:
    """Crawl the target URL using a DefaultMarkdownGenerator configured with the specified content source."""
    # Configure the markdown generator:
    #
    # content_source selects the HTML input fed into the markdown pipeline:
    #   "raw_html"     — the full, unmodified HTML returned by the browser,
    #                    including navigation, footers, ads, and script tags
    #   "cleaned_html" — Crawl4AI's pre-processed HTML with common boilerplate
    #                    (scripts, styles, hidden elements) stripped out
    #   "fit_html"     — the most aggressively filtered version, keeping only
    #                    the main content block most relevant to the page topic
    #
    # options tweak the markdown rendering behaviour:
    #   "ignore_links" — drop hyperlink syntax from the output so the markdown
    #                    reads as plain prose rather than being cluttered with URLs
    #   "body_width"   — wrap long lines at this column width (78 chars here) to
    #                    keep the output terminal-friendly
    generator = DefaultMarkdownGenerator(
        content_source=source,
        options={"ignore_links": True, "body_width": 78},
    )

    # Build the crawler run configuration:
    # - BYPASS cache so we always fetch the live page instead of stale data
    # - Attach our custom markdown generator to override the default pipeline
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, markdown_generator=generator, verbose=False
    )

    # Launch an async crawler session using a context manager, which handles browser startup and teardown
    async with AsyncWebCrawler() as crawler:
        # Perform the actual crawl — arun() fetches the page, renders any
        # JavaScript, and converts the visible content into markdown using our configured generator
        result = await crawler.arun(url=URL, config=config)

    # Extract the raw markdown string from the result object and display the
    # source label, total character count, and a 140-character preview
    markdown = result.markdown
    # raw_markdown is the unfiltered output from the selected content_source
    markdown_text = markdown.raw_markdown

    markdown_preview = markdown_text[:140]
    print()
    print(source, "chars=", len(markdown_text), "preview=")
    print(markdown_preview)


############################## Main Entry Logic ##############################
async def main() -> None:
    """Run the same crawl three times, once for each supported content source."""
    # Cycle through each content source variant in order from least to most
    # filtered, printing a comparative summary line for each
    for source in ("raw_html", "cleaned_html", "fit_html"):
        await crawl_with_source(source)


################################# Entry Point ################################
if __name__ == "__main__":
    asyncio.run(main())
