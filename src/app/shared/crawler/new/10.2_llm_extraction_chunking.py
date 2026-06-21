import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

############################# Path Configuration #############################

# When running this script directly (e.g., `python 10.2_llm_extraction_chunking.py`),
# __package__ will be None or empty. In that case, add the project root (two levels
# up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
    LLMExtractionStrategy,
)

from crawl4ai_101.common.io import load_env

# Target URL — the Crawl4AI quickstart documentation page
URL = "https://docs.crawl4ai.com/core/quickstart/"

############################ Output Schema Definition ########################


# Pydantic model that defines the structured JSON shape we want the LLM to
# produce. Crawl4AI serialises this into a JSON Schema and passes it to the
# LLM so the model knows exactly which fields to populate.
#
# - title      : the heading of the page
# - key_steps  : an ordered list of the main quickstart steps described on the page
class PageSummary(BaseModel):
    title: str = ""
    key_steps: list[str] = Field(default_factory=list)


############################ Main Crawl Routine ##############################


async def main() -> None:
    """Crawl a page and extract structured data from it using an LLM strategy.

    Demonstrates how to pair LLMExtractionStrategy with chunking so that long
    pages are split into overlapping token-sized windows before being sent to
    the model. The LLM is instructed to fill a Pydantic-backed JSON Schema,
    guaranteeing a predictable output structure regardless of page length.
    """
    # Load environment variables from a .env file (e.g. OPENAI_API_KEY)
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required for LLM extraction.")
        return

    # Build the LLM extraction strategy — this tells Crawl4AI how to send
    # page content to an LLM and parse the response back into structured data
    strategy = LLMExtractionStrategy(
        # Which LLM to call and how to authenticate with it
        llm_config=LLMConfig(provider="openai/gpt-4o-mini", api_token=api_key),
        # Pass the JSON Schema derived from our Pydantic model so the LLM
        # knows the exact field names and types it must populate
        schema=PageSummary.model_json_schema(),
        # extraction_type="schema" tells the strategy to validate and coerce
        # the LLM response against the provided JSON Schema (as opposed to
        # "block" mode which returns free-form labelled text blocks)
        extraction_type="schema",
        # Natural-language prompt appended to each chunk sent to the LLM
        instruction="Extract the page title and 3 to 5 quickstart steps as JSON.",
        # input_format="fit_markdown" converts the crawled HTML into a compact
        # markdown representation before passing it to the LLM, reducing token
        # usage compared to sending raw HTML
        input_format="fit_markdown",
        # ── Chunking parameters ──────────────────────────────────────────────
        # chunk_token_threshold: maximum number of tokens per chunk; content
        # exceeding this limit is split into multiple LLM calls
        chunk_token_threshold=1400,
        # overlap_rate: fraction of each chunk that is repeated at the start of
        # the next chunk (0.1 = 10% overlap) so context is not lost at split points
        overlap_rate=0.1,
        # apply_chunking: set to True to enable the chunking behaviour described
        # above; set to False to send the entire page in a single LLM call
        apply_chunking=True,
        # extra_args are passed directly to the underlying LLM API call,
        # allowing fine-grained control over sampling (temperature) and the
        # maximum number of tokens the model may generate per chunk (max_tokens)
        extra_args={"temperature": 0.2, "max_tokens": 700},
    )

    # Attach the extraction strategy to the crawler run configuration and
    # bypass the cache so we always fetch a fresh copy of the page
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, extraction_strategy=strategy, verbose=False)

    # Run the crawl inside an async context manager — the browser is launched
    # on entry and shut down cleanly on exit
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=URL, config=config)

    # Guard against crawl or extraction failures before accessing the payload
    if not result.success:
        print("llm extraction failed:", result.error_message)
        return

    # Parse the JSON string stored in extracted_content back into a Python
    # dict/list and print it; fall back to an empty object if the field is None
    print(json.loads(result.extracted_content or "{}"))

    # Print a token-usage summary (prompt tokens, completion tokens, cost) for
    # all LLM calls made during this extraction run — useful for cost tracking
    strategy.show_usage()


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
