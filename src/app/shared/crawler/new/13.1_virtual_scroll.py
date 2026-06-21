"""Video 14: Virtual scrolling with VirtualScrollConfig.

Demonstrates:
- a self-contained virtualized feed
- VirtualScrollConfig on a scrolling container
- merging replaced DOM content into one result

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_14_virtual_scroll.py`
"""

import asyncio
import json

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, JsonCssExtractionStrategy, VirtualScrollConfig

RAW_VIRTUAL_FEED = """
<html>
  <body>
    <div id="feed" style="height:320px; overflow:auto; border:1px solid #ccc;">
      <div id="spacer" style="height:7200px; position:relative;">
        <div id="viewport" style="position:absolute; left:0; right:0;"></div>
      </div>
    </div>
    <script>
      const items = Array.from({length: 120}, (_, i) => `Feed Item ${i + 1}`);
      const feed = document.getElementById("feed");
      const viewport = document.getElementById("viewport");
      const rowHeight = 60;
      function render() {
        const start = Math.floor(feed.scrollTop / rowHeight);
        const visible = items.slice(start, start + 8);
        viewport.style.transform = `translateY(${start * rowHeight}px)`;
        viewport.innerHTML = visible.map((item, index) => `
          <article class="feed-item">
            <h2>${item}</h2>
            <p>Virtual row ${start + index + 1}</p>
          </article>
        `).join("");
      }
      feed.addEventListener("scroll", render);
      render();
    </script>
  </body>
</html>
"""


async def main() -> None:
    schema = {
        "name": "Virtual Feed",
        "baseSelector": "article.feed-item",
        "fields": [
            {"name": "title", "selector": "h2", "type": "text"},
            {"name": "body", "selector": "p", "type": "text"},
        ],
    }
    config = CrawlerRunConfig(
        virtual_scroll_config=VirtualScrollConfig(
            container_selector="#feed",
            scroll_count=12,
            scroll_by="container_height",
            wait_after_scroll=0.1,
        ),
        extraction_strategy=JsonCssExtractionStrategy(schema),
        wait_for="css:#feed",
        verbose=False,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(f"raw://{RAW_VIRTUAL_FEED}", config=config)

    if not result.success:
        print(f"Virtual scroll failed: {result.error_message}")
        return

    items = json.loads(result.extracted_content or "[]")
    print(f"Virtual items captured: {len(items)}")
    if items:
        print(f"First item: {items[0]}")
        print(f"Last item: {items[-1]}")


if __name__ == "__main__":
    asyncio.run(main())
