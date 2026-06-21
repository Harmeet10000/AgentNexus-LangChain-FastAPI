"""Video 19: Deep crawling with BFS, DFS, and BestFirst strategies.

Demonstrates:
- BFS and BestFirst deep crawling
- filter chains and keyword scoring
- streaming vs non-streaming deep crawl results

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_19_deep_crawling.py`
"""

import asyncio

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import (
    BFSDeepCrawlStrategy,
    BestFirstCrawlingStrategy,
    FilterChain,
    KeywordRelevanceScorer,
    URLPatternFilter,
)

ROOT_URL = "https://docs.crawl4ai.com/"


async def main() -> None:
    bfs_config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=1,
            max_pages=5,
            include_external=False,
            filter_chain=FilterChain([URLPatternFilter(["*core/*", "*api/*", "*quickstart*"])]),
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False,
    )
    best_first_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=1,
            max_pages=5,
            include_external=False,
            filter_chain=FilterChain([URLPatternFilter(["*core/*", "*api/*", "*quickstart*"])]),
            url_scorer=KeywordRelevanceScorer(
                keywords=["browser", "crawler", "markdown", "config"],
                weight=0.7,
            ),
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=True,
        verbose=False,
    )

    async with AsyncWebCrawler() as crawler:
        bfs_results = await crawler.arun(ROOT_URL, config=bfs_config)
        streamed = []
        async for result in await crawler.arun(ROOT_URL, config=best_first_config):
            streamed.append(result)

    print(f"BFS results: {len(bfs_results)}")
    for result in bfs_results[:3]:
        depth = (result.metadata or {}).get("depth", 0)
        print(f"BFS page: depth={depth} url={result.url}")

    print(f"BestFirst streamed results: {len(streamed)}")
    for result in streamed[:3]:
        metadata = result.metadata or {}
        print(
            f"BestFirst page: depth={metadata.get('depth', 0)} "
            f"score={metadata.get('score', 0):.2f} url={result.url}"
        )
    print("Crash recovery and prefetch are doc-level follow-ups when your installed version supports them.")


if __name__ == "__main__":
    asyncio.run(main())
