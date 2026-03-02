"""Shared crawler module."""

from app.shared.crawler.chunker import (
    Chunk,
    extract_headers,
    extract_title_from_markdown,
    smart_chunk_markdown,
    truncate_content,
)
from app.shared.crawler.config import CrawlerConfig, get_crawler_config
from app.shared.crawler.crawler import CrawlResult, WebCrawler, get_crawler
from app.shared.crawler.processor import (
    ExtractionResult,
    GeminiProcessor,
    SchemaType,
    get_processor,
    get_schema_for_type,
)
from app.shared.crawler.validator import (
    get_domain_from_url,
    is_valid_url,
    sanitize_url,
    validate_url,
)

__all__ = [
    "Chunk",
    "CrawlResult",
    "CrawlerConfig",
    "ExtractionResult",
    "GeminiProcessor",
    "SchemaType",
    "WebCrawler",
    "extract_headers",
    "extract_title_from_markdown",
    "get_crawler",
    "get_crawler_config",
    "get_domain_from_url",
    "get_processor",
    "get_schema_for_type",
    "is_valid_url",
    "sanitize_url",
    "smart_chunk_markdown",
    "truncate_content",
    "validate_url",
]
