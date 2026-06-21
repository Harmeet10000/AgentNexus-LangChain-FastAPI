"""Video 16: Hooks and authentication.

Demonstrates:
- on_page_context_created for login and resource control
- before_goto for extra headers
- before_retrieve_html for final page interaction

Prerequisites:
- `pip install crawl4ai playwright`
- `playwright install`

Run:
- `python crawl4ai_101/video_16_hooks_and_auth.py`
"""

import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

LOGIN_URL = "https://the-internet.herokuapp.com/login"
SECURE_URL = "https://the-internet.herokuapp.com/secure"
USERNAME = "tomsmith"
PASSWORD = "SuperSecretPassword!"


async def main() -> None:
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(wait_for="css:.flash", verbose=False)

    async def on_page_context_created(page, context, **kwargs):
        await page.route(
            "**/*.{png,jpg,jpeg,gif,svg,webp}",
            lambda route: route.abort(),
        )
        await page.goto(LOGIN_URL)
        await page.fill("#username", USERNAME)
        await page.fill("#password", PASSWORD)
        await page.click("button[type='submit']")
        return page

    async def before_goto(page, context, url, **kwargs):
        await page.set_extra_http_headers({"X-Crawl4AI-Demo": "hooks"})
        return page

    async def before_retrieve_html(page, context, **kwargs):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        return page

    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawler.crawler_strategy.set_hook("on_page_context_created", on_page_context_created)
        crawler.crawler_strategy.set_hook("before_goto", before_goto)
        crawler.crawler_strategy.set_hook("before_retrieve_html", before_retrieve_html)
        result = await crawler.arun(SECURE_URL, config=run_config)

    if not result.success:
        print(f"Hook-based auth failed: {result.error_message}")
        return

    success = "You logged into a secure area!" in (result.cleaned_html or "")
    print(f"Hook-based auth success: {success}")
    print("Hooks used: on_page_context_created, before_goto, before_retrieve_html")


if __name__ == "__main__":
    asyncio.run(main())
