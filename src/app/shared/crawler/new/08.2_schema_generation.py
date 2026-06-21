"""Video 08: Auto-generate extraction schemas with an LLM.

Demonstrates:
- JsonCssExtractionStrategy.generate_schema()
- saving a generated schema for reuse
- running the generated schema without another LLM call

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`
- `OPENAI_API_KEY` set in your environment

Run:
- `python crawl4ai_101/video_08_schema_generation.py`
"""

import asyncio
import json
import os
from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
    LLMConfig,
)

API_KEY = os.getenv("OPENAI_API_KEY")
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "video_08"
SAMPLE_HTML = """
<section class="products">
  <article class="product-card">
    <h2>Starter Drone</h2>
    <p class="price">$249.00</p>
    <p class="rating">4.7 stars</p>
  </article>
  <article class="product-card">
    <h2>Studio Microphone</h2>
    <p class="price">$189.00</p>
    <p class="rating">4.9 stars</p>
  </article>
</section>
"""


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


async def main() -> None:
    if not API_KEY:
        print("SKIP: set OPENAI_API_KEY to generate a schema with an LLM.")
        return
    if not hasattr(JsonCssExtractionStrategy, "generate_schema"):
        print("SKIP: this Crawl4AI version does not expose generate_schema().")
        return

    output_dir = ensure_output_dir()
    schema = JsonCssExtractionStrategy.generate_schema(
        SAMPLE_HTML,
        query="Extract each product name, price, and rating as a flat record.",
        llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token=API_KEY),
    )
    schema_path = output_dir / "product_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(schema),
        verbose=False,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(f"raw://{SAMPLE_HTML}", config=config)

    if not result.success:
        print(f"Generated-schema crawl failed: {result.error_message}")
        return

    items = json.loads(result.extracted_content or "[]")
    print(f"Generated schema saved to: {schema_path}")
    print(f"Reusable extraction rows: {len(items)}")
    if items:
        print(f"First row: {items[0]}")


if __name__ == "__main__":
    asyncio.run(main())
