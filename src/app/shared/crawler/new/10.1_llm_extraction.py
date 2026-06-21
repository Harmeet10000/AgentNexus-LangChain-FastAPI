"""Video 10: LLMExtractionStrategy for complex semantic extraction.

Demonstrates:
- schema-based LLM extraction with Pydantic
- fit_markdown as a lower-token input format
- token usage reporting with show_usage()

Prerequisites:
- `pip install crawl4ai playwright pydantic`
- `playwright install`
- `OPENAI_API_KEY` set in your environment

Run:
- `python crawl4ai_101/video_10_llm_extraction.py`
"""

import asyncio
import json
import os

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    LLMConfig,
    LLMExtractionStrategy,
    PruningContentFilter,
)

API_KEY = os.getenv("OPENAI_API_KEY")
RAW_PAGE = """
<main>
  <h1>Creator Gear Guide</h1>
  <p>The FocusCaster camera costs $799 and records clean 4K video for solo creators.</p>
  <p>The SoundLift microphone costs $149 and is tuned for voice-overs and podcasts.</p>
  <p>The BeamPad light kit costs $99 and includes two diffused LED panels.</p>
</main>
"""

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None
    Field = None


if BaseModel:
    class Product(BaseModel):
        name: str = Field(..., description="Product name")
        price: str = Field(..., description="Displayed price")
        description: str = Field(..., description="Short explanation of the product")


async def main() -> None:
    if not API_KEY:
        print("SKIP: set OPENAI_API_KEY to run the LLM extraction demo.")
        return
    if BaseModel is None:
        print("SKIP: install pydantic to run the schema-based extraction demo.")
        return

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.3, threshold_type="fixed")
    )
    strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token=API_KEY),
        schema=Product.model_json_schema(),
        extraction_type="schema",
        instruction="Extract every product with its name, price, and description.",
        input_format="fit_markdown",
        chunk_token_threshold=1000,
        extra_args={"temperature": 0, "max_tokens": 800},
    )
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=markdown_generator,
        extraction_strategy=strategy,
        verbose=False,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(f"raw://{RAW_PAGE}", config=config)

    if not result.success:
        print(f"LLM extraction failed: {result.error_message}")
        return

    items = json.loads(result.extracted_content or "[]")
    print(f"Products extracted: {len(items)}")
    if items:
        print(f"First product: {items[0]}")
    if hasattr(strategy, "show_usage"):
        strategy.show_usage()


if __name__ == "__main__":
    asyncio.run(main())
