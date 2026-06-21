"""Video 06: Understanding every field in CrawlResult.

Demonstrates:
- markdown, html, cleaned_html, links, media
- screenshot, PDF, and MHTML capture
- saving output artifacts to the local output folder

Prerequisites:
- `pip install crawl4ai playwright`
- `crawl4ai-setup` to install the required browsers for visual capture
"""

import asyncio
import base64
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from rich import print

################################ Configuration ################################
# Target page used to demonstrate multi-format capture (markdown, image, PDF, MHTML)
# URL = "https://en.wikipedia.org/wiki/Web_scraping"
URL = "https://en.wikipedia.org/wiki/MHTML#External_links"


# Resolve output directory relative to this script for portable artifact storage
local_path = "output/video_06"
OUTPUT_DIR = Path(__file__).resolve().parent / local_path

################################ Helper Functions ################################
def ensure_output_dir() -> Path:
    """Create the output directory tree if missing and return its path."""
    # Use parents=True so nested folders are created in a single call
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_bytes(path: Path, data: bytes | None) -> None:
    """Write binary data to disk, skipping silently when no data is provided."""
    # Guard against None/empty payloads so callers don't need to pre-check
    if data:
        path.write_bytes(data)


################################ Main Crawl Routine ################################
async def main() -> None:
    """Crawl the target URL and persist markdown plus visual artifacts."""
    output_dir = ensure_output_dir()

    # Request every supported artifact in a single crawl pass
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # Force a fresh fetch instead of using cached results
        screenshot=True,  # Capture a full-page PNG screenshot
        pdf=False,  # Render the page to a PDF document
        capture_mhtml=True,  # Save a single-file MHTML archive of the page
        verbose=False,
    )

    # Use the async context manager so the underlying browser is cleaned up automatically
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL, config=config)

    ############################ Persist Artifacts ############################
    # Prefer the raw markdown attribute, falling back to string form for older result shapes
    markdown = getattr(result.markdown, "raw_markdown", str(result.markdown))
    # Write markdown to disk as UTF-8 text;
    (output_dir / "page.md").write_text(markdown or "", encoding="utf-8")

    # Screenshots arrive as base64; decode before writing to a binary PNG
    if result.screenshot:
        # The result.screenshot field is a base64-encoded string representing PNG image data.
        png_bytes = base64.b64decode(result.screenshot)
        save_bytes(output_dir / "page.png", png_bytes)

    # PDF bytes are already binary and may be None when rendering is unavailable
    if result.pdf:
        save_bytes(output_dir / "page.pdf", result.pdf)

    # MHTML is plain text and only present when capture_mhtml succeeded
    if result.mhtml:
        (output_dir / "page.mhtml").write_text(result.mhtml, encoding="utf-8")

    ############################ Crawl Summary ############################
    # Surface quick stats so the tutorial viewer can verify the crawl worked
    internal_links = len((result.links or {}).get("internal", []))
    external_links = len((result.links or {}).get("external", []))
    image_count = len((result.media or {}).get("images", []))
    # Show all available fields in the CrawlResult for this page
    print(result[0].__dict__.keys())

    print(f"\nSaved markdown to: '{local_path}/page.md'")
    print(f"\nInternal links: {internal_links}, external links: {external_links}")
    print(f"\nImages discovered: {image_count}")
    print("\nArtifacts available:")
    print(f"png={bool(result.screenshot)} pdf={bool(result.pdf)} mhtml={bool(result.mhtml)}\n")


################################ Entry Point ################################
# Run the async pipeline only when executed directly, not on import
if __name__ == "__main__":
    asyncio.run(main())
