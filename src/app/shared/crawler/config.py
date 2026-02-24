"""Crawler configuration and settings."""

from dataclasses import dataclass
from typing import Any

from app.config.settings import get_settings


@dataclass
class CrawlerConfig:
    """Configuration for the web crawler."""

    headless: bool = True
    timeout: int = 30000
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    max_depth: int = 3
    max_pages: int = 10
    max_content_size: int = 102400
    max_concurrent: int = 10
    memory_threshold: float = 70.0
    cache_mode: str = "bypass"

    proxy_server: str | None = None
    proxy_enabled: bool = False

    def __post_init__(self):
        settings = get_settings()
        self.headless = settings.CRAWL4AI_HEADLESS
        self.timeout = settings.CRAWL4AI_TIMEOUT
        self.user_agent = settings.CRAWL4AI_USER_AGENT
        self.max_depth = settings.CRAWL4AI_MAX_DEPTH
        self.max_pages = settings.CRAWL4AI_MAX_PAGES
        self.max_content_size = settings.CRAWL4AI_MAX_CONTENT_SIZE
        self.proxy_server = settings.CRAWL4AI_PROXY
        self.proxy_enabled = settings.CRAWL4AI_PROXY_ENABLED

    def get_proxy_dict(self) -> dict[str, Any] | None:
        """Get proxy configuration for Crawl4AI."""
        if self.proxy_enabled and self.proxy_server:
            return {"server": self.proxy_server}
        return None

    def to_browser_config(self) -> dict[str, Any]:
        """Convert to Crawl4AI BrowserConfig kwargs."""
        config = {
            "headless": self.headless,
            "timeout": self.timeout,
            "extra_args": [
                f"--user-agent={self.user_agent}",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        }

        proxy = self.get_proxy_dict()
        if proxy:
            config["proxy"] = proxy

        return config

    def to_crawler_run_config(self) -> dict[str, Any]:
        """Convert to Crawl4AI CrawlerRunConfig kwargs."""
        return {
            "cache_mode": self.cache_mode,
            "stream": False,
        }


def get_crawler_config() -> CrawlerConfig:
    """Get crawler configuration from settings."""
    return CrawlerConfig()
