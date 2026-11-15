"""HTTP-compatible re-exports of crawler application contracts.

The wire shapes remain unchanged, while ownership lives in the application
layer so workers and adapters never import the HTTP transport package.
"""

from app.features.crawler.application.contracts import (
    CrawlJobChunk,
    CrawlJobChunkList,
    CrawlJobPageList,
    CrawlJobSearchResponse,
    CrawlRequest,
    CrawlResponse,
    RateLimitInfo,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

__all__ = [
    "CrawlJobChunk",
    "CrawlJobChunkList",
    "CrawlJobPageList",
    "CrawlJobSearchResponse",
    "CrawlRequest",
    "CrawlResponse",
    "RateLimitInfo",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
]
