"""Video 04: Content selection with css_selector and target_elements.

Demonstrates:
- css_selector for scoped extraction
- target_elements for markdown focus with broader page context
- excluded_tags to strip entire tag types (e.g. nav, footer)
- excluded_selector to strip elements matching a CSS selector
- feature detection for newer options like flatten_shadow_dom

Prerequisites:
- `pip install crawl4ai`
- `crawl4ai-setup`
"""

################# Imports & Constants #################
import asyncio
import inspect
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from rich import print

URL = "https://docs.crawl4ai.com/core/quickstart/"


################# Main Crawl Logic #################
async def main() -> None:
    """Run two crawls to compare css_selector vs target_elements extraction."""

    # Limit extraction to content inside the <main> element only
    scoped_config = CrawlerRunConfig(
        # css_selector="main",
        css_selector="#mkdocs-terminal-content > ol:nth-child(3)",
        delay_before_return_html=0.5,  # Allow dynamic content to render
        verbose=False,
    )

    # Extract markdown from multiple elements while preserving full page context
    focused_config = CrawlerRunConfig(
        # target_elements=["main", "nav"],
        target_elements=["main"],
        delay_before_return_html=0.5,
        verbose=False,
    )

    # Drop entire tag types from the parsed HTML before markdown conversion
    excluded_tags_config = CrawlerRunConfig(
        excluded_tags=["nav", "footer", "aside"],
        delay_before_return_html=0.5,
        verbose=False,
    )

    # Drop any element matching a CSS selector (e.g. cookie banners, ads)
    excluded_selector_config = CrawlerRunConfig(
        excluded_selector=".md-sidebar, .md-footer, main",
        delay_before_return_html=0.5,
        verbose=False,
    )

    # Run both crawl strategies against the same URL
    async with AsyncWebCrawler() as crawler:
        scoped = await crawler.arun(URL, config=scoped_config)
        focused = await crawler.arun(URL, config=focused_config)
        excluded_tags = await crawler.arun(URL, config=excluded_tags_config)
        excluded_selector = await crawler.arun(URL, config=excluded_selector_config)

    ################# Compare Results #################
    # Markdown length and a preview of the extracted content for each strategy
    print(f"\n########## Content selection results for {URL} ##########")
    # Include results for css_selector and target_elements
    if scoped.success:
        len_links = len(scoped.links.get("internal", []))
        print(f"css_selector markdown={len(scoped.markdown)} internal_links={len_links}")
        print(scoped.markdown[:700])

    if focused.success:
        len_links = len(focused.links.get("internal", []))
        print(f"\ntarget_elements markdown={len(focused.markdown)} internal_links={len_links}")
        print(focused.markdown[:700])

    ################# Exclusion Results #################
    if excluded_tags.success:
        print("\nexcluded_tags=['nav','footer','aside']")
        print(f"markdown={len(excluded_tags.markdown)}")
        print(excluded_tags.markdown[:700])

    if excluded_selector.success:
        print("\nexcluded_selector='.md-sidebar, .md-footer'")
        print(f"markdown={len(excluded_selector.markdown)}")
        print(excluded_selector.markdown[:700])


if __name__ == "__main__":
    asyncio.run(main())
