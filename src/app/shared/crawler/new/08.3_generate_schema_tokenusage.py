import asyncio
import json
import os
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python 08.3_generate_schema_tokenusage.py`),
# __package__ will be None or empty. In that case, add the project root (two levels up)
# to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
    LLMConfig,
)

from crawl4ai_101.common.io import SCHEMA_CACHE_DIR, load_env, write_json

# Target URL — the Crawl4AI quickstart documentation page we want to extract
# structured data from
URL = "https://docs.crawl4ai.com/core/quickstart/"

# Path where the LLM-generated schema will be cached on disk so we avoid
# paying LLM API costs on every run (generate once, reuse forever)
SCHEMA_FILE = SCHEMA_CACHE_DIR / "v11_quickstart_schema.json"

############################ HTML Sampling Helper ############################


async def fetch_sample_html() -> str:
    """Fetch a truncated HTML snippet from the target URL for schema generation.

    The LLM only needs a representative slice of the page to infer CSS selectors —
    sending the full document would waste tokens and may exceed model context limits.
    We cap the output at 12 000 characters, preferring cleaned HTML (boilerplate
    removed) over the raw source when Crawl4AI makes it available.
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=URL,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=False),
        )
    if not result.success:
        raise RuntimeError(result.error_message)

    # Prefer cleaned_html (tags stripped, noise removed) so the LLM sees a
    # compact, signal-rich view of the page structure. Fall back to raw html
    # if cleaned_html is unavailable. Truncate to 12 000 chars to stay within
    # typical LLM context limits while still covering the page's key sections.
    return (result.cleaned_html or result.html or "")[:12000]


################################ Main Routine ################################


async def main() -> None:
    """Generate a CSS extraction schema via an LLM, then use it to scrape the page.

    The workflow has two phases:

    1. Schema generation (runs once):
       - Fetch a sample of the page HTML.
       - Pass it to JsonCssExtractionStrategy.generate_schema(), which sends
         the HTML and a plain-English query to an LLM.  The LLM inspects the
         markup and returns a JSON schema describing the CSS selectors and
         field names needed to extract the requested data.
       - Persist the schema to disk so future runs skip the LLM call entirely.

    2. Extraction (runs every time):
       - Load the cached schema and build a JsonCssExtractionStrategy from it.
       - Run a fresh crawl of the page using that strategy to extract structured
         JSON content — no LLM involved at this stage.
    """
    # Load environment variables from a .env file (e.g., OPENAI_API_KEY)
    load_env()

    # Schema generation requires an OpenAI API key — bail out early with a
    # clear message rather than producing a cryptic auth error later
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required for schema generation.")
        return

    # Ensure the schema cache directory exists before attempting to write to it
    SCHEMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if SCHEMA_FILE.exists():
        # Schema already generated on a previous run — load it from disk to
        # avoid an unnecessary (and billable) LLM API call
        schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
        print("using cached schema:", SCHEMA_FILE)
    else:
        # Ask the LLM to analyse a sample of the page HTML and produce a
        # JSON schema containing CSS selectors for the requested fields.
        # LLMConfig provider format: "<provider>/<model>" — here we use
        # OpenAI's gpt-4o-mini, which balances quality and low token cost.
        schema = JsonCssExtractionStrategy.generate_schema(
            html=await fetch_sample_html(),
            schema_type="CSS",
            query="Extract the article title and major quickstart sections from this page.",
            llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token=api_key),
        )
        # Persist the generated schema so subsequent runs skip the LLM step
        write_json(SCHEMA_FILE, schema)
        print("generated schema:", SCHEMA_FILE)

    # Build an extraction strategy from the (possibly cached) schema — this
    # applies the CSS selectors deterministically, with no further LLM calls
    strategy = JsonCssExtractionStrategy(schema, verbose=False)

    # Combine the extraction strategy with a fresh (non-cached) crawl config
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, extraction_strategy=strategy, verbose=False)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=URL, config=config)

    # Report crawl outcome and a short preview of the structured JSON payload
    print("success:", result.success)
    print("extracted preview:", (result.extracted_content or "")[:250])


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
