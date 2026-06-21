"""Video 07: Structured extraction with JsonCssExtractionStrategy.

Demonstrates:
- CSS schema extraction on a live site
- attribute extraction and base element attributes
- nested extraction on raw HTML

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_07_json_css_extraction.py`
"""

import asyncio
import json

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, JsonCssExtractionStrategy

HN_URL = "https://news.ycombinator.com/"
RAW_CATALOG_HTML = """
<section class="catalog">
  <div class="product" data-sku="kb-001">
    <h2 class="name">Mechanical Keyboard</h2>
    <span class="price">$129.00</span>
    <div class="reviews">
      <div class="review"><span class="author">Ava</span><span class="rating">5</span></div>
      <div class="review"><span class="author">Noah</span><span class="rating">4</span></div>
    </div>
  </div>
  <div class="product" data-sku="ms-002">
    <h2 class="name">Vertical Mouse</h2>
    <span class="price">$89.00</span>
    <div class="reviews">
      <div class="review"><span class="author">Mia</span><span class="rating">5</span></div>
    </div>
  </div>
</section>
"""


async def extract_hn() -> None:
    schema = {
        "name": "Hacker News Stories",
        "baseSelector": "tr.athing",
        "baseFields": [{"name": "story_id", "type": "attribute", "attribute": "id"}],
        "fields": [
            {"name": "rank", "selector": "span.rank", "type": "text", "transform": "strip"},
            {
                "name": "title",
                "selector": "span.titleline > a",
                "type": "text",
                "transform": "strip",
            },
            {
                "name": "url",
                "selector": "span.titleline > a",
                "type": "attribute",
                "attribute": "href",
            },
        ],
    }
    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(schema, verbose=False),
        css_selector="tr.athing",
        verbose=False,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(HN_URL, config=config)
    if not result.success:
        print(f"Live CSS extraction failed: {result.error_message}")
        return
    items = json.loads(result.extracted_content or "[]")
    print(f"Live stories extracted: {len(items)}")
    if items:
        print(f"First story: {items[0]}")


async def extract_nested_catalog() -> None:
    schema = {
        "name": "Catalog",
        "baseSelector": "div.product",
        "baseFields": [{"name": "sku", "type": "attribute", "attribute": "data-sku"}],
        "fields": [
            {"name": "name", "selector": "h2.name", "type": "text"},
            {"name": "price", "selector": "span.price", "type": "text"},
            {
                "name": "reviews",
                "selector": "div.review",
                "type": "nested_list",
                "fields": [
                    {"name": "author", "selector": "span.author", "type": "text"},
                    {"name": "rating", "selector": "span.rating", "type": "text"},
                ],
            },
        ],
    }
    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(schema, verbose=False),
        verbose=False,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(f"raw://{RAW_CATALOG_HTML}", config=config)
    items = json.loads(result.extracted_content or "[]")
    print(f"Nested products extracted: {len(items)}")
    if items:
        print(f"First product: {items[0]}")


async def main() -> None:
    await extract_hn()
    await extract_nested_catalog()


if __name__ == "__main__":
    asyncio.run(main())
