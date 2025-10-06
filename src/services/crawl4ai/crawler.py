"""Web crawling service using Crawl4AI and alternative scrapers."""

from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import asyncio
import hashlib

from bs4 import BeautifulSoup
import httpx
from playwright.async_api import async_playwright
from markdownify import markdownify as md

from src.core.config.settings import settings
from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import redis_cache

logger = LoggerAdapter(__name__)


class WebCrawler:
    """Service for web crawling and content extraction."""

    def __init__(self):
        """Initialize web crawler."""
        self.headless = settings.crawl4ai_headless
        self.timeout = settings.crawl4ai_timeout
        self.user_agent = settings.crawl4ai_user_agent
        self.browser = None
        self.playwright = None

    async def initialize(self):
        """Initialize Playwright browser."""
        if not self.playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=self.headless)

    async def cleanup(self):
        """Clean up browser resources."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def crawl_url(
        self,
        url: str,
        max_depth: int = 1,
        extract_links: bool = True,
        extract_images: bool = False,
        use_cache: bool = True,
        javascript: bool = False,
    ) -> Dict[str, Any]:
        """Crawl a URL and extract content."""
        try:
            # Check cache
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            cache_key = f"crawl:url:{url_hash}"

            if use_cache:
                cached_result = await redis_cache.get(cache_key)
                if cached_result:
                    logger.info("Returning cached crawl result")
                    return cached_result

            # Choose crawling method
            if javascript:
                result = await self._crawl_with_playwright(
                    url, extract_links, extract_images
                )
            else:
                result = await self._crawl_with_httpx(
                    url, extract_links, extract_images
                )

            # Crawl deeper if requested
            if max_depth > 1 and result.get("links"):
                result["child_pages"] = []
                for link in result["links"][:5]:  # Limit to 5 child pages
                    child_result = await self.crawl_url(
                        link,
                        max_depth=max_depth - 1,
                        extract_links=False,
                        extract_images=False,
                        use_cache=use_cache,
                        javascript=javascript,
                    )
                    result["child_pages"].append(child_result)

            # Cache result
            if use_cache:
                await redis_cache.set(cache_key, result, ttl=3600)

            logger.info(f"Crawled URL: {url}")
            return result

        except Exception as e:
            logger.error(f"Failed to crawl URL: {url}", error=str(e))
            return {
                "url": url,
                "error": str(e),
                "content": "",
                "title": "",
                "links": [],
                "images": [],
            }

    async def _crawl_with_httpx(
        self, url: str, extract_links: bool, extract_images: bool
    ) -> Dict[str, Any]:
        """Crawl URL using httpx (for static content)."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout / 1000,  # Convert to seconds
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract content
            result = {
                "url": url,
                "status_code": response.status_code,
                "title": soup.title.string if soup.title else "",
                "content": self._extract_text_content(soup),
                "markdown": md(str(soup)),
                "links": [],
                "images": [],
                "metadata": {
                    "content_type": response.headers.get("content-type", ""),
                    "content_length": len(response.content),
                },
            }

            # Extract links
            if extract_links:
                result["links"] = self._extract_links(soup, url)

            # Extract images
            if extract_images:
                result["images"] = self._extract_images(soup, url)

            return result

    async def _crawl_with_playwright(
        self, url: str, extract_links: bool, extract_images: bool
    ) -> Dict[str, Any]:
        """Crawl URL using Playwright (for JavaScript-rendered content)."""
        await self.initialize()

        page = await self.browser.new_page(user_agent=self.user_agent)

        try:
            # Navigate to page
            await page.goto(url, wait_until="networkidle", timeout=self.timeout)

            # Wait for content to load
            await page.wait_for_load_state("domcontentloaded")

            # Get page content
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            # Extract content
            result = {
                "url": url,
                "title": await page.title(),
                "content": self._extract_text_content(soup),
                "markdown": md(str(soup)),
                "links": [],
                "images": [],
                "metadata": {
                    "viewport": await page.viewport_size(),
                    "url_final": page.url,
                },
            }

            # Extract links
            if extract_links:
                result["links"] = self._extract_links(soup, url)

            # Extract images
            if extract_images:
                result["images"] = self._extract_images(soup, url)

            # Take screenshot
            screenshot = await page.screenshot()
            result["screenshot"] = screenshot

            return result

        finally:
            await page.close()

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract text content from BeautifulSoup object."""
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from page."""
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            # Only include HTTP/HTTPS links
            if absolute_url.startswith(("http://", "https://")):
                links.append(absolute_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links

    def _extract_images(
        self, soup: BeautifulSoup, base_url: str
    ) -> List[Dict[str, str]]:
        """Extract all images from page."""
        images = []
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src:
                absolute_url = urljoin(base_url, src)
                images.append(
                    {
                        "src": absolute_url,
                        "alt": img.get("alt", ""),
                        "title": img.get("title", ""),
                    }
                )
        return images

    async def crawl_sitemap(
        self, sitemap_url: str, max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """Crawl pages from sitemap."""
        try:
            # Fetch sitemap
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    sitemap_url, headers={"User-Agent": self.user_agent}, timeout=30
                )
                response.raise_for_status()

            # Parse sitemap
            soup = BeautifulSoup(response.text, "xml")
            urls = []

            for loc in soup.find_all("loc"):
                urls.append(loc.text)
                if len(urls) >= max_pages:
                    break

            # Crawl URLs
            results = []
            for url in urls:
                result = await self.crawl_url(url, max_depth=1, javascript=False)
                results.append(result)

            logger.info(f"Crawled {len(results)} pages from sitemap")
            return results

        except Exception as e:
            logger.error("Failed to crawl sitemap", error=str(e))
            return []

    async def extract_structured_data(
        self, url: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract structured data from page based on schema."""
        try:
            # Crawl page
            page_data = await self.crawl_url(url, javascript=True)

            if not page_data.get("content"):
                return {}

            # Parse content with BeautifulSoup
            soup = BeautifulSoup(page_data.get("markdown", ""), "html.parser")

            # Extract data based on schema
            extracted_data = {}
            for field, selector in schema.items():
                if isinstance(selector, str):
                    element = soup.select_one(selector)
                    extracted_data[field] = element.text.strip() if element else None
                elif isinstance(selector, dict):
                    # Complex extraction
                    selector_type = selector.get("type", "text")
                    css_selector = selector.get("selector", "")

                    if selector_type == "list":
                        elements = soup.select(css_selector)
                        extracted_data[field] = [el.text.strip() for el in elements]
                    elif selector_type == "attribute":
                        element = soup.select_one(css_selector)
                        attr_name = selector.get("attribute", "href")
                        extracted_data[field] = (
                            element.get(attr_name) if element else None
                        )
                    else:
                        element = soup.select_one(css_selector)
                        extracted_data[field] = (
                            element.text.strip() if element else None
                        )

            return extracted_data

        except Exception as e:
            logger.error("Failed to extract structured data", error=str(e))
            return {}


# Create global instance
web_crawler = WebCrawler()
