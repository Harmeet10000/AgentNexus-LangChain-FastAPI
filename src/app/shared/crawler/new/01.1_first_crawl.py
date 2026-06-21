"""Video 01: Your first crawl in 5 lines of Python.

Demonstrates:
- AsyncWebCrawler and arun()
- CrawlResult success, markdown, html, cleaned_html, status_code
- Minimal error handling

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_01_first_crawl.py`
"""

import asyncio
from textwrap import shorten

from crawl4ai import AsyncWebCrawler

URL = "https://example.com"
PREVIEW_CHARS = 320


def preview(text: str, width: int = PREVIEW_CHARS) -> str:
    return shorten(" ".join((text or "").split()), width=width, placeholder="...")

async def main() -> None:
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL)

    print(f"URL: {result.url}")
    print(f"Success: {result.success}")
    print(f"Status code: {result.status_code}")

    if not result.success:
        print(f"Error: {result.error_message}")
        return

    markdown = getattr(result.markdown, "raw_markdown", str(result.markdown))
    print(f"Markdown preview: {preview(markdown)}")
    print(f"HTML chars: {len(result.html or '')}")
    print(f"Cleaned HTML chars: {len(result.cleaned_html or '')}")
    print(f"Raw HTML preview: {preview(result.html)}")


if __name__ == "__main__":
    asyncio.run(main())
