"""
4-crawl_and_chunk_markdown.py
-----------------------------
Scrapes a Markdown (.md or .txt) page using Crawl4AI, then splits the content into chunks based on # and ## headers.
Prints each chunk for further processing or inspection.
Usage: Set the target URL in main(), then run as a script.
"""
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
import re

async def scrape_and_chunk_markdown(url: str):
    """
    Scrape a Markdown page and split into chunks by # and ## headers.
    """
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig()
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        if not result.success:
            print(f"Failed to crawl {url}: {result.error_message}")
            return
        markdown = result.markdown
        # Split by headers (#, ##)
        # Find all # and ## headers to use as chunk boundaries
        header_pattern = re.compile(r'^(# .+|## .+)$', re.MULTILINE)
        headers = [m.start() for m in header_pattern.finditer(markdown)] + [len(markdown)]
        chunks = []
        # Split the markdown into chunks between headers
        for i in range(len(headers)-1):
            chunk = markdown[headers[i]:headers[i+1]].strip()
            if chunk:
                chunks.append(chunk)
        print(f"Split into {len(chunks)} chunks:")
        for idx, chunk in enumerate(chunks):
            print(f"\n--- Chunk {idx+1} ---\n{chunk}\n")

if __name__ == "__main__":
    url = "https://ai.pydantic.dev/llms-full.txt"
    asyncio.run(scrape_and_chunk_markdown(url))
