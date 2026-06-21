import asyncio
import sys
from pathlib import Path

############################# Path Configuration #############################

# When running this script directly (e.g., `python v19_url_seeding.py`), __package__
# will be None or empty. In that case, add the project root (two levels up) to
# sys.path so that sibling package imports resolve correctly.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from crawl4ai import AsyncUrlSeeder, AsyncWebCrawler, CacheMode, CrawlerRunConfig, SeedingConfig

# The domain to seed — AsyncUrlSeeder will discover URLs belonging to this domain
# without fully crawling every page (lightweight alternative to a full-site crawl)
DOMAIN = "docs.crawl4ai.com"

############################### Main Routine #################################


async def main() -> None:
    """Demonstrate the two-phase URL seeding and crawling pattern.

    Phase 1 — Seed: use AsyncUrlSeeder to discover a list of URLs for a domain
    by reading its XML sitemap.  This is much faster than crawling every link
    because it reads structured metadata rather than rendering full pages.

    Phase 2 — Crawl: pass the validated URL list to AsyncWebCrawler.arun_many()
    so all pages are fetched concurrently in a single batched operation.
    """
    ############################ Phase 1: Seed URLs ##########################

    # AsyncUrlSeeder discovers URLs for a domain without performing a full crawl.
    # It supports multiple sources (sitemap, crawl, etc.) and enriches each
    # discovered URL with metadata such as HTTP status and <head> tag content.
    async with AsyncUrlSeeder() as seeder:
        seeded = await seeder.urls(
            DOMAIN,
            SeedingConfig(
                # source="sitemap" reads the domain's XML sitemap file(s) —
                # the fastest way to get a structured list of all published URLs
                source="sitemap",
                # pattern filters the discovered URLs using glob-style matching;
                # only URLs containing "docs.crawl4ai.com" are kept
                pattern="*docs.crawl4ai.com*",
                # max_urls caps the total number of URLs returned so this demo
                # stays fast and doesn't hammer the server
                max_urls=25,
                # extract_head=True fetches the <head> section of each URL to
                # populate metadata fields (title, description, etc.) in the results
                extract_head=True,
            ),
        )

    # Each entry in `seeded` is a dict with at minimum a "url" key and a "status"
    # field.  Filter to only those marked "valid" (i.e., 2xx responses, not
    # redirects or errors), then take the first 5 to keep the demo output brief.
    valid_urls = [row["url"] for row in seeded if row.get("status") == "valid"][:5]
    print("seeded:", len(seeded), "valid:", len(valid_urls))

    ########################### Phase 2: Crawl URLs ##########################

    # Open a single crawler session and crawl all valid URLs concurrently.
    # arun_many() is more efficient than calling arun() in a loop because it
    # manages a shared browser pool across all requests.
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(
            valid_urls,
            config=CrawlerRunConfig(
                # BYPASS cache so every URL is fetched live rather than served
                # from a previous crawl stored on disk
                cache_mode=CacheMode.BYPASS,
                verbose=False,
            ),
        )

    # Report how many pages were crawled successfully out of the total attempted
    print("crawled:", sum(1 for result in results if result.success), "of", len(results))


################################# Entry Point ################################

# Standard Python entry-point guard — use asyncio.run() to execute the
# async main() coroutine from a synchronous context
if __name__ == "__main__":
    asyncio.run(main())
