"""Crawler configuration and settings."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings


class CrawlerConfig(BaseModel):
    """Configuration for the web crawler."""

    model_config = ConfigDict(extra="forbid")

    headless: bool = Field(default_factory=lambda: get_settings().CRAWL4AI_HEADLESS)
    timeout: int = Field(default_factory=lambda: get_settings().CRAWL4AI_TIMEOUT)
    user_agent: str = Field(default_factory=lambda: get_settings().CRAWL4AI_USER_AGENT)
    max_depth: int = Field(default_factory=lambda: get_settings().CRAWL4AI_MAX_DEPTH)
    max_pages: int = Field(default_factory=lambda: get_settings().CRAWL4AI_MAX_PAGES)
    max_content_size: int = Field(default_factory=lambda: get_settings().CRAWL4AI_MAX_CONTENT_SIZE)
    max_concurrent: int = 10
    memory_threshold: float = 70.0
    cache_mode: str = "bypass"

    proxy_server: str | None = Field(default_factory=lambda: get_settings().CRAWL4AI_PROXY)
    proxy_enabled: bool = Field(default_factory=lambda: get_settings().CRAWL4AI_PROXY_ENABLED)

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
