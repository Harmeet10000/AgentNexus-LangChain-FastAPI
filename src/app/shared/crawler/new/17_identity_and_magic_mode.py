"""Video 17: Identity-based crawling and magic mode.

Demonstrates:
- managed browser profiles with CRAWL4AI_PROFILE_DIR
- BrowserProfiler discovery
- magic mode with locale, timezone, and geolocation

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`
- Optional: set `CRAWL4AI_PROFILE_DIR` to a saved Crawl4AI profile path

Run:
- `python crawl4ai_101/video_17_identity_and_magic_mode.py`
"""

import asyncio
import os
from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    BrowserProfiler,
    CrawlerRunConfig,
    GeolocationConfig,
)

PROFILE_DIR = os.getenv("CRAWL4AI_PROFILE_DIR")
URL = "https://example.com"


async def run_managed_browser() -> None:
    if not PROFILE_DIR or not Path(PROFILE_DIR).exists():
        print("Managed browser demo skipped: set CRAWL4AI_PROFILE_DIR to a valid profile path.")
        return

    browser_config = BrowserConfig(
        use_managed_browser=True,
        user_data_dir=PROFILE_DIR,
        browser_type="chromium",
        headless=True,
        verbose=False,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(URL)
    print(f"Managed browser success: {result.success} profile={PROFILE_DIR}")


async def run_magic_mode() -> None:
    config = CrawlerRunConfig(
        magic=True,
        remove_overlay_elements=True,
        locale="de-DE",
        timezone_id="Europe/Berlin",
        geolocation=GeolocationConfig(latitude=52.52, longitude=13.405),
        verbose=False,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(URL, config=config)
    print(f"Magic mode success: {result.success} locale=de-DE timezone=Europe/Berlin")


async def main() -> None:
    profiles = BrowserProfiler().list_profiles()
    print(f"Profiles discovered by BrowserProfiler: {len(profiles)}")
    await run_managed_browser()
    await run_magic_mode()


if __name__ == "__main__":
    asyncio.run(main())
