"""Video 15: Session management for multi-page workflows.

Demonstrates:
- session_id reuse across multiple arun() calls
- js_only on follow-up page interactions
- accumulating paginated results

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_15_session_management.py`
"""

import asyncio
import json

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, JsonCssExtractionStrategy

URL = "https://github.com/microsoft/TypeScript/commits/main"
SESSION_ID = "video_15_commits"

COMMIT_SCHEMA = {
    "name": "Commits",
    "baseSelector": "li[data-testid='commit-row-item']",
    "fields": [
        {"name": "title", "selector": "h4 a", "type": "text", "transform": "strip"},
        {"name": "href", "selector": "h4 a", "type": "attribute", "attribute": "href"},
    ],
}

NEXT_PAGE_JS = """
const rows = document.querySelectorAll('li[data-testid="commit-row-item"] h4');
if (rows.length > 0) {
  window.firstCommitTitle = rows[0].textContent.trim();
}
document.querySelector('a[data-testid="pagination-next-button"]')?.click();
"""

WAIT_FOR_CHANGE = """
js:() => {
  const rows = document.querySelectorAll('li[data-testid="commit-row-item"] h4');
  if (!rows.length) return false;
  return rows[0].textContent.trim() !== window.firstCommitTitle;
}
"""


async def main() -> None:
    all_commits = []
    strategy = JsonCssExtractionStrategy(COMMIT_SCHEMA, verbose=False)

    async with AsyncWebCrawler() as crawler:
        for page in range(2):
            config = CrawlerRunConfig(
                session_id=SESSION_ID,
                extraction_strategy=strategy,
                css_selector="li[data-testid='commit-row-item']",
                js_code=NEXT_PAGE_JS if page else None,
                wait_for=WAIT_FOR_CHANGE if page else "css:li[data-testid='commit-row-item']",
                js_only=page > 0,
                cache_mode=CacheMode.BYPASS,
                verbose=False,
            )
            result = await crawler.arun(URL, config=config)
            if not result.success:
                print(f"Page {page + 1} failed: {result.error_message}")
                break
            page_commits = json.loads(result.extracted_content or "[]")
            all_commits.extend(page_commits)
            print(f"Page {page + 1} commits: {len(page_commits)}")
        await crawler.crawler_strategy.kill_session(SESSION_ID)

    print(f"Accumulated commits: {len(all_commits)}")
    if all_commits:
        print(f"First commit sample: {all_commits[0]}")


if __name__ == "__main__":
    asyncio.run(main())
