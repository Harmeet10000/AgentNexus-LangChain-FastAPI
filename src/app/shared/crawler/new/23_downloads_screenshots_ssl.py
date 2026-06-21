"""Video 23: Downloads, screenshots, PDFs, and SSL inspection.

Demonstrates:
- file download handling with accept_downloads
- screenshot, PDF, and MHTML capture
- SSL certificate export from CrawlResult

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_23_downloads_screenshots_ssl.py`
"""

import asyncio
import base64
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "video_23"
DOWNLOAD_PAGE_URL = "https://www.python.org/downloads/windows/"
CAPTURE_URL = "https://example.com/"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


async def download_demo(output_dir: Path) -> None:
    downloads_dir = output_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    browser_config = BrowserConfig(
        accept_downloads=True,
        downloads_path=str(downloads_dir),
        headless=True,
        verbose=False,
    )
    run_config = CrawlerRunConfig(
        js_code='document.querySelector(\'a[href$="embed-amd64.zip"]\')?.click();',
        wait_for="css:body",
        delay_before_return_html=4.0,
        verbose=False,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(DOWNLOAD_PAGE_URL, config=run_config)
    if result.downloaded_files:
        print(f"Downloaded files: {result.downloaded_files}")
    else:
        print("Download demo completed, but no downloaded files were reported.")


async def capture_demo(output_dir: Path) -> None:
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        screenshot=True,
        screenshot_wait_for=1.0,
        pdf=True,
        capture_mhtml=True,
        fetch_ssl_certificate=True,
        verbose=False,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(CAPTURE_URL, config=run_config)

    if not result.success:
        print(f"Capture crawl failed: {result.error_message}")
        return

    if result.screenshot:
        (output_dir / "capture.png").write_bytes(base64.b64decode(result.screenshot))
    if result.pdf:
        (output_dir / "capture.pdf").write_bytes(result.pdf)
    if result.mhtml:
        (output_dir / "capture.mhtml").write_text(result.mhtml, encoding="utf-8")
    if result.ssl_certificate:
        result.ssl_certificate.to_json(output_dir / "certificate.json")
        print(
            "SSL certificate: "
            f"issuer={result.ssl_certificate.issuer} valid_until={result.ssl_certificate.valid_until}"
        )
    else:
        print("SSL certificate was not available in this environment.")

    print(
        "Artifacts saved: "
        f"png={bool(result.screenshot)} pdf={bool(result.pdf)} mhtml={bool(result.mhtml)}"
    )


async def main() -> None:
    output_dir = ensure_output_dir()
    await download_demo(output_dir)
    await capture_demo(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
