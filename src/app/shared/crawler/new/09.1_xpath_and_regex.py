"""Video 09: XPath extraction and regex extraction.

Demonstrates:
- JsonXPathExtractionStrategy on a table
- RegexExtractionStrategy with built-in patterns
- custom regex patterns for prices and SKUs

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_09_xpath_and_regex.py`
"""

import asyncio
import json

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    JsonXPathExtractionStrategy,
    RegexExtractionStrategy,
)

TABLE_HTML = """
<table id="releases">
  <tr><th>Name</th><th>Version</th><th>Date</th></tr>
  <tr><td>Python</td><td>3.13.1</td><td>2024-12-03</td></tr>
  <tr><td>Playwright</td><td>1.49.0</td><td>2024-12-10</td></tr>
</table>
"""

CONTACT_HTML = """
<main>
  <p>Email support@example.com or sales@example.org for help.</p>
  <p>Visit https://docs.crawl4ai.com/ for the docs.</p>
  <p>Starter plan: $19.00, Pro plan: $79.00, SKU KB-200.</p>
</main>
"""


async def run_xpath_demo() -> None:
    schema = {
        "name": "Release Table",
        "baseSelector": "//table[@id='releases']//tr[position()>1]",
        "fields": [
            {"name": "name", "selector": "./td[1]", "type": "text"},
            {"name": "version", "selector": "./td[2]", "type": "text"},
            {"name": "date", "selector": "./td[3]", "type": "text"},
        ],
    }
    config = CrawlerRunConfig(
        extraction_strategy=JsonXPathExtractionStrategy(schema),
        verbose=False,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(f"raw://{TABLE_HTML}", config=config)
    rows = json.loads(result.extracted_content or "[]")
    print(f"XPath rows: {rows}")


async def run_regex_demo() -> None:
    strategy = RegexExtractionStrategy(
        pattern=(
            RegexExtractionStrategy.Email
            | RegexExtractionStrategy.Url
            | RegexExtractionStrategy.Currency
        ),
        custom={
            "sku": r"\b[A-Z]{2}-\d{3}\b",
            "usd_price": r"\$\d+(?:\.\d{2})?",
        },
    )
    config = CrawlerRunConfig(extraction_strategy=strategy, verbose=False)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(f"raw://{CONTACT_HTML}", config=config)
    matches = json.loads(result.extracted_content or "[]")
    print(f"Regex matches found: {len(matches)}")
    print(f"Sample matches: {matches[:5]}")


async def main() -> None:
    await run_xpath_demo()
    await run_regex_demo()


if __name__ == "__main__":
    asyncio.run(main())
