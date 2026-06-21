"""Video 12: Choose and combine extraction strategies.

Demonstrates:
- a CSS-first then regex pipeline
- optional semantic clustering with CosineStrategy
- optional chunking strategies when the local version supports them

Prerequisites:
- `pip install crawl4ai playwright sentence-transformers`
- `playwright install`

Run:
- `python crawl4ai_101/video_12_strategy_selection.py`
"""

import asyncio
import json

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CosineStrategy,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
    RegexChunking,
    RegexExtractionStrategy,
)

try:
    from crawl4ai.chunking_strategy import OverlappingWindowChunking
except ImportError:
    OverlappingWindowChunking = None

RAW_PRODUCTS = """
<section class="products">
  <article class="card">
    <h2 class="name">Compact Camera</h2>
    <p class="description">Weight 420g. Size 12 x 7 x 5 cm. Built for travel creators.</p>
  </article>
  <article class="card">
    <h2 class="name">Desk Microphone</h2>
    <p class="description">Weight 860g. Size 18 x 9 x 9 cm. Great for podcasts and voiceovers.</p>
  </article>
</section>
"""
SEMANTIC_URL = "https://docs.crawl4ai.com/core/quickstart/"


async def run_css_then_regex() -> None:
    css_schema = {
        "name": "Products",
        "baseSelector": "article.card",
        "fields": [
            {"name": "name", "selector": "h2.name", "type": "text"},
            {"name": "description", "selector": "p.description", "type": "text"},
        ],
    }
    css_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=JsonCssExtractionStrategy(css_schema),
        verbose=False,
    )
    regex_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=RegexExtractionStrategy(
            custom={
                "weight_grams": r"\b\d{3}g\b",
                "dimensions_cm": r"\b\d{1,2} x \d{1,2} x \d{1,2} cm\b",
            }
        ),
        verbose=False,
    )

    async with AsyncWebCrawler() as crawler:
        css_result = await crawler.arun(f"raw://{RAW_PRODUCTS}", config=css_config)
        regex_result = await crawler.arun(f"raw://{RAW_PRODUCTS}", config=regex_config)

    products = json.loads(css_result.extracted_content or "[]")
    matches = json.loads(regex_result.extracted_content or "[]")
    print(f"Decision tree: structured HTML -> CSS, fine-grained patterns -> Regex")
    print(f"CSS products: {products}")
    print(f"Regex matches: {matches}")


async def run_cosine_demo() -> None:
    try:
        chunking = OverlappingWindowChunking(window_size=500, overlap=50) if OverlappingWindowChunking else RegexChunking()
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=CosineStrategy(
                semantic_filter="browser config markdown extraction",
                top_k=3,
            ),
            chunking_strategy=chunking,
            verbose=False,
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(SEMANTIC_URL, config=config)
        print(
            "Optional semantic pass: "
            f"success={result.success} extracted_len={len(result.extracted_content or '')}"
        )
    except Exception as exc:
        print(f"Optional CosineStrategy demo skipped: {exc}")


async def main() -> None:
    await run_css_then_regex()
    await run_cosine_demo()


if __name__ == "__main__":
    asyncio.run(main())
