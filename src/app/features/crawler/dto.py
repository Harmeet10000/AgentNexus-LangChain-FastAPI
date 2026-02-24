"""Crawler feature DTOs (Data Transfer Objects)."""

from typing import Any

from pydantic import BaseModel, Field

from app.features.crawler.constants import CrawlMode, SchemaType


class CrawlRequest(BaseModel):
    """Request to crawl a URL."""

    url: str = Field(..., description="URL to crawl")
    mode: CrawlMode = Field(default=CrawlMode.MARKDOWN, description="Output mode")
    max_depth: int = Field(
        default=1, ge=1, le=5, description="Recursion depth (1 = single page)"
    )
    max_pages: int = Field(
        default=10, ge=1, le=50, description="Max pages for recursive crawl"
    )
    use_proxy: bool = Field(default=False, description="Use proxy for crawling")
    bypass_cache: bool = Field(default=False, description="Bypass cache")
    extract_structured: bool = Field(
        default=False, description="Extract structured data"
    )
    schema_type: SchemaType | None = Field(
        default=None, description="Predefined schema type"
    )
    custom_schema: dict[str, Any] | None = Field(
        default=None, description="Custom JSON schema"
    )
    summary: bool = Field(default=False, description="Generate summary using Gemini")
    timeout: int = Field(default=30, ge=5, le=120, description="Timeout in seconds")


class CrawlResultItem(BaseModel):
    """Single crawl result."""

    url: str
    success: bool
    title: str | None = None
    markdown: str | None = None
    html: str | None = None
    summary: str | None = None
    extracted_data: dict[str, Any] | None = None
    word_count: int | None = None
    crawl_time_ms: int | None = None
    cached: bool = False
    error_message: str | None = None
    links: list[str] = Field(default_factory=list)


class CrawlResponse(BaseModel):
    """Response from crawl operation."""

    success: bool
    query_url: str
    results: list[CrawlResultItem]
    total_pages: int
    successful_pages: int
    failed_pages: int
    total_word_count: int
    processing_time_ms: int


class SearchRequest(BaseModel):
    """Request to search the web."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    max_results: int = Field(default=10, ge=1, le=20, description="Max results")
    include_answer: bool = Field(default=True, description="Include AI answer")


class SearchResultItem(BaseModel):
    """Single search result."""

    url: str
    title: str
    content: str
    score: float
    published_date: str | None = None


class SearchResponse(BaseModel):
    """Response from search operation."""

    success: bool
    query: str
    answer: str | None = None
    results: list[SearchResultItem]
    total_results: int


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    remaining_minute: int
    remaining_hour: int
