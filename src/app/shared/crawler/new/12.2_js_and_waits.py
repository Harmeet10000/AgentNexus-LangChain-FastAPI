"""Video 13: JavaScript execution and wait conditions.

Demonstrates:
- js_code for clicks and scrolling
- CSS and JavaScript wait_for conditions
- reusing a session to load more content

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_13_js_and_waits.py`
"""

import asyncio
import json

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, JsonCssExtractionStrategy

URL = "https://news.ycombinator.com/"
SESSION_ID = "video_13_hn"
STORY_SCHEMA = {
    "name": "Stories",
    "baseSelector": "tr.athing",
    "fields": [{"name": "title", "selector": "span.titleline > a", "type": "text"}],
}


def story_summary(result) -> tuple[int, str]:
    data = json.loads(result.extracted_content or "[]")
    first_title = data[0]["title"] if data else "n/a"
    return len(data), first_title


async def main() -> None:
    initial_config = CrawlerRunConfig(
        session_id=SESSION_ID,
        wait_for="css:tr.athing:nth-of-type(10)",
        extraction_strategy=JsonCssExtractionStrategy(STORY_SCHEMA),
        cache_mode=CacheMode.BYPASS,
        verbose=False,
    )
    load_more_config = CrawlerRunConfig(
        session_id=SESSION_ID,
        js_code=[
            "window.scrollTo(0, document.body.scrollHeight);",
            "document.querySelector('a.morelink')?.click();",
        ],
        wait_for="js:() => document.querySelectorAll('tr.athing').length > 30",
        extraction_strategy=JsonCssExtractionStrategy(STORY_SCHEMA),
        js_only=True,
        cache_mode=CacheMode.BYPASS,
        verbose=False,
    )

    async with AsyncWebCrawler() as crawler:
        first = await crawler.arun(URL, config=initial_config)
        second = await crawler.arun(URL, config=load_more_config)
        await crawler.crawler_strategy.kill_session(SESSION_ID)

    if not first.success:
        print(f"Initial crawl failed: {first.error_message}")
        return
    if not second.success:
        print(f"Interactive crawl failed: {second.error_message}")
        return

    first_count, first_title = story_summary(first)
    second_count, second_title = story_summary(second)
    print(f"Initial page stories: {first_count} first_title={first_title}")
    print(f"After JS click: {second_count} first_title={second_title}")
    print("If your Crawl4AI version adds js_code_before_wait, this is where it fits.")


if __name__ == "__main__":
    asyncio.run(main())
