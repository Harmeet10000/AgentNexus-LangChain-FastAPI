import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from rich import print

############################ Configuration ###################################

# Target URL to crawl — Wikipedia page rich in links and media for comparison
URL = "https://meta.wikimedia.org/wiki/VideoWiki"
URL = "https://docs.crawl4ai.com/"
# Alternate URL with more media (YouTube) for testing filter effects:
# URL = "https://www.youtube.com/watch?v=fzld3WjvBSk"

############################ Helper: Summarize ###############################
def summarize(label: str, result) -> None:
    """Print a one-line summary of link and image counts from a crawl result.

    Args:
        label:  Short identifier printed at line start (e.g. "base", "trimmed").
        result: CrawlResult from AsyncWebCrawler.arun().
    """
    # Safely unpack link buckets, falling back to empty lists when missing
    internal = (result.links or {}).get("internal", [])
    external = (result.links or {}).get("external", [])

    # Safely unpack media buckets by type for comparison reporting
    images = (result.media or {}).get("images", [])
    audios = (result.media or {}).get("audios", [])
    videos = (result.media or {}).get("videos", [])

    # Emit a single-line comparison row for easy side-by-side reading
    print(
        label,
        "html_chars=",
        len(result.html or ""),  # Rough page-size proxy
        "internal_links=",
        len(internal),
        "external_links=",
        len(external),
        "images=",
        len(images),
        "audios=",
        len(audios),
        "videos=",
        len(videos),
    )


############################ Main Crawl Routine ##############################
async def main() -> None:
    """Run two crawls of the same URL and compare link/media counts.

    "base"    — default settings; captures all links and images.
    "trimmed" — applies filters:
        exclude_external_images:    drops off-domain <img> (CDNs, trackers)
        exclude_social_media_links: drops social-media hrefs (X, FB, LinkedIn...)

    Comparing both shows how much external/social clutter a page carries.
    """
    # Use async context manager to ensure browser resources are cleaned up
    async with AsyncWebCrawler() as crawler:
        # Baseline crawl — bypass cache to guarantee a fresh fetch for fair comparison
        base = await crawler.arun(
            url=URL,
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=False),
        )

        # Filtered crawl — same URL but with noise-reduction flags enabled
        trimmed = await crawler.arun(
            url=URL,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                exclude_external_images=True,  # Strip off-domain <img> tags (CDNs, trackers)
                exclude_social_media_links=True,  # Strip links to X, Facebook, LinkedIn, etc.
                verbose=False,
            ),
        )

    # Only summarize successful crawls to avoid AttributeErrors on failure
    if base.success:
        summarize("\nBase:\n", base)
    if trimmed.success:
        summarize("\nTrimmed:\n", trimmed)


################################# Entry Point ################################

# Bridge sync entry point into the async event loop
if __name__ == "__main__":
    asyncio.run(main())
