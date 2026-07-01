"""Crawler configuration and settings."""

from typing import Any

from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
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

    # SPEC-01: Content noise filtering
    excluded_tags: list[str] = Field(
        default_factory=lambda: get_settings().CRAWL4AI_EXCLUDED_TAGS.split(",")
    )
    word_count_threshold: int = Field(
        default_factory=lambda: get_settings().CRAWL4AI_WORD_COUNT_THRESHOLD
    )
    pruning_threshold: float = Field(
        default_factory=lambda: get_settings().CRAWL4AI_PRUNING_THRESHOLD
    )
    ignore_links: bool = True
    ignore_images: bool = True

    # SPEC-03: Viewport & stealth
    viewport_width: int = Field(default_factory=lambda: get_settings().CRAWL4AI_VIEWPORT_WIDTH)
    viewport_height: int = Field(default_factory=lambda: get_settings().CRAWL4AI_VIEWPORT_HEIGHT)
    enable_stealth: bool = Field(default_factory=lambda: get_settings().CRAWL4AI_STEALTH)
    user_agent_mode: str = "fixed"

    # SPEC-04: Wait strategy
    page_timeout: int = Field(default_factory=lambda: get_settings().CRAWL4AI_PAGE_TIMEOUT)
    wait_until: str | None = Field(default_factory=lambda: get_settings().CRAWL4AI_WAIT_UNTIL)
    wait_for: str | None = Field(default_factory=lambda: get_settings().CRAWL4AI_WAIT_FOR)

    # SPEC-07: Rate limiting
    rate_limit_delay: tuple[float, float] = Field(
        default_factory=lambda: (
            get_settings().CRAWL4AI_RATE_LIMIT_DELAY_MIN,
            get_settings().CRAWL4AI_RATE_LIMIT_DELAY_MAX,
        )
    )

    # SPEC-08: URL-specific configs
    url_patterns: list[str] = Field(default_factory=list)

    # Magic mode: locale, timezone, geolocation spoofing
    magic: bool = Field(default_factory=lambda: get_settings().CRAWL4AI_MAGIC)
    locale: str | None = Field(default_factory=lambda: get_settings().CRAWL4AI_LOCALE)
    timezone_id: str | None = Field(default_factory=lambda: get_settings().CRAWL4AI_TIMEZONE_ID)
    geolocation_lat: float | None = Field(default_factory=lambda: get_settings().CRAWL4AI_GEO_LAT)
    geolocation_lon: float | None = Field(default_factory=lambda: get_settings().CRAWL4AI_GEO_LON)

    # Monitor
    enable_monitor: bool = Field(default_factory=lambda: get_settings().CRAWL4AI_ENABLE_MONITOR)

    def get_proxy_dict(self) -> dict[str, Any] | None:
        """Get proxy configuration for Crawl4AI."""
        if self.proxy_enabled and self.proxy_server:
            return {"server": self.proxy_server}
        return None

    def to_browser_config(self) -> dict[str, Any]:
        """Convert to Crawl4AI BrowserConfig kwargs."""
        config: dict[str, Any] = {
            "headless": self.headless,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "extra_args": [
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        }

        if self.user_agent_mode == "random":
            config["user_agent_mode"] = "random"
        else:
            config["extra_args"].append(f"--user-agent={self.user_agent}")

        if self.enable_stealth:
            config["enable_stealth"] = True

        proxy = self.get_proxy_dict()
        if proxy:
            config["proxy"] = proxy

        return config

    def to_crawler_run_config(self) -> dict[str, Any]:
        """Convert to Crawl4AI CrawlerRunConfig kwargs."""
        config: dict[str, Any] = {
            "cache_mode": self.cache_mode,
            "stream": False,
            "word_count_threshold": self.word_count_threshold,
            "excluded_tags": self.excluded_tags,
            "exclude_external_links": True,
            "exclude_social_media_links": True,
        }
        if self.page_timeout:
            config["page_timeout"] = self.page_timeout
        if self.wait_until:
            config["wait_until"] = self.wait_until
        if self.wait_for:
            config["wait_for"] = self.wait_for
        if self.magic:
            config["magic"] = True
            config["remove_overlay_elements"] = True
        if self.locale:
            config["locale"] = self.locale
        if self.timezone_id:
            config["timezone_id"] = self.timezone_id
        if self.geolocation_lat is not None and self.geolocation_lon is not None:
            config["geolocation"] = {
                "latitude": self.geolocation_lat,
                "longitude": self.geolocation_lon,
            }
        return config

    def get_markdown_generator(
        self,
        content_filter: Any | None = None,
    ) -> DefaultMarkdownGenerator:
        """Get DefaultMarkdownGenerator with configured options and optional content filter."""
        options = {
            "ignore_links": self.ignore_links,
            "ignore_images": self.ignore_images,
        }

        if content_filter is None:
            content_filter = PruningContentFilter(
                threshold=self.pruning_threshold,
                threshold_type="fixed",
                min_word_threshold=self.word_count_threshold,
            )

        return DefaultMarkdownGenerator(content_filter=content_filter, options=options)


def get_crawler_config() -> CrawlerConfig:
    """Get crawler configuration from settings."""
    return CrawlerConfig()
