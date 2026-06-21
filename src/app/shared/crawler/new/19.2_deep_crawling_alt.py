import asyncio
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v18_deep_crawling.py`),
# __package__ will be None or empty. In that case, add the project root (two
# levels up) to sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import (
    AsyncWebCrawler,
    BestFirstCrawlingStrategy,
    CrawlerRunConfig,
    FilterChain,
    KeywordRelevanceScorer,
    URLPatternFilter,
)

# Define the seed URL where the deep crawl begins
START_URL = "https://docs.crawl4ai.com/"

############################ Main Crawl Routine ##############################


async def main() -> None:
    """Perform a streaming deep crawl using best-first URL prioritization.

    This demonstrates Crawl4AI's deep crawling capability: starting from a
    single seed URL, the crawler discovers and follows links automatically,
    scoring each candidate URL by keyword relevance and visiting the most
    promising pages first. Results are streamed back as each page completes
    rather than waiting for the entire crawl to finish.
    """
    # BestFirstCrawlingStrategy explores discovered URLs in order of their
    # relevance score, always visiting the highest-scoring unvisited URL next
    # (similar to a best-first / greedy search). This is more efficient than
    # a naive BFS when only a subset of pages is actually relevant.
    #
    # - max_depth: maximum number of link hops from the start URL (depth 0).
    #   A value of 1 means the crawler will visit the seed page and any pages
    #   linked directly from it, but will not follow links on those child pages.
    # - max_pages: hard cap on the total number of pages crawled across all
    #   depths, regardless of how many links are discovered.
    # - include_external: when False, links pointing to domains other than the
    #   seed domain are ignored entirely.
    strategy = BestFirstCrawlingStrategy(
        max_depth=1,
        max_pages=6,
        include_external=False,
        # KeywordRelevanceScorer assigns a numeric score to each candidate URL
        # based on how many of the provided keywords appear in it. The weight
        # parameter controls how heavily the keyword signal influences the final
        # score relative to other heuristics (e.g., URL depth).
        url_scorer=KeywordRelevanceScorer(keywords=["crawl", "config", "quickstart"], weight=0.7),
        # FilterChain applies a sequence of filters to each discovered URL;
        # a URL must pass every filter to be enqueued for crawling.
        # URLPatternFilter accepts glob-style patterns and rejects any URL that
        # does not match at least one pattern — here we restrict crawling to
        # pages on the docs.crawl4ai.com subdomain only.
        filter_chain=FilterChain([URLPatternFilter(patterns=["*docs.crawl4ai.com*"])]),
    )

    # deep_crawl_strategy wires the strategy into the run config so arun()
    # knows to follow links instead of fetching a single page.
    # stream=True tells arun() to return an async iterator that yields each
    # CrawlResult as soon as that individual page finishes, rather than
    # collecting all results and returning them together at the end.
    config = CrawlerRunConfig(deep_crawl_strategy=strategy, stream=True, verbose=False)

    async with AsyncWebCrawler() as crawler:
        # Because stream=True is set, arun() returns an async generator;
        # we iterate over it to process pages one by one as they arrive.
        stream = await crawler.arun(START_URL, config=config)
        async for result in stream:
            # result.metadata is a dict populated by the deep crawl strategy.
            # It carries per-page context that is not part of the raw crawl
            # output, including:
            #   "depth" — how many hops from the start URL this page is
            #   "score" — the relevance score assigned by the URL scorer
            metadata = result.metadata or {}
            print(
                "depth=",
                metadata.get("depth", 0),
                "score=",
                metadata.get("score", 0.0),
                "url=",
                result.url,
            )


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
