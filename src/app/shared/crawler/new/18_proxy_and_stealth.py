"""Video 18: Proxy rotation and anti-bot stealth.

Demonstrates:
- enable_stealth on the browser config
- simulate_user and magic mode on the run config
- proxy rotation with PROXIES and RoundRobinProxyStrategy

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`
- Optional: set `PROXIES` to test proxy rotation

Run:
- `python crawl4ai_101/video_18_proxy_and_stealth.py`
"""

import asyncio
import os
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, ProxyConfig
from crawl4ai.proxy_strategy import RoundRobinProxyStrategy

STEALTH_URL = "https://example.com"
IP_CHECK_URL = "https://httpbin.org/ip"
PROXIES = os.getenv("PROXIES")


async def run_stealth_demo() -> None:
    browser_config = BrowserConfig(enable_stealth=True, headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        simulate_user=True,
        magic=True,
        verbose=False,
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(STEALTH_URL, config=run_config)
    print(f"Stealth crawl success: {result.success}")
    print("Undetected-browser example is a documented variant when your setup supports it.")


async def run_proxy_rotation_demo() -> None:
    if not PROXIES:
        print("Proxy rotation skipped: set PROXIES to test rotating proxies.")
        return

    proxies = ProxyConfig.from_env()
    if not proxies:
        print("Proxy rotation skipped: PROXIES was set but no valid proxies were parsed.")
        return

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        proxy_rotation_strategy=RoundRobinProxyStrategy(proxies),
        verbose=False,
    )
    async with AsyncWebCrawler(config=BrowserConfig(headless=True, verbose=False)) as crawler:
        results = await crawler.arun_many([IP_CHECK_URL] * min(len(proxies), 3), config=run_config)

    for index, result in enumerate(results, start=1):
        ip_match = re.search(r"(?:\d{1,3}\.){3}\d{1,3}", result.html or "")
        detected_ip = ip_match.group(0) if ip_match else "unknown"
        print(f"Proxy request {index}: success={result.success} ip={detected_ip}")


async def main() -> None:
    await run_stealth_demo()
    await run_proxy_rotation_demo()


if __name__ == "__main__":
    asyncio.run(main())
