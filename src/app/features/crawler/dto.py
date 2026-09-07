"""Crawler feature DTOs (Data Transfer Objects)."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.features.crawler.constants import CrawlMode, SchemaType


class CrawlRequest(BaseModel):
    """Request to crawl a URL."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=4096, description="URL to crawl")
    mode: CrawlMode = Field(default=CrawlMode.MARKDOWN, description="Output mode")
    max_depth: int = Field(default=1, ge=1, le=5, description="Recursion depth (1 = single page)")
    max_pages: int = Field(default=10, ge=1, le=50, description="Max pages for recursive crawl")
    use_proxy: bool = Field(default=False, description="Use proxy for crawling")
    bypass_cache: bool = Field(default=False, description="Bypass cache")
    extract_structured: bool = Field(default=False, description="Extract structured data")
    schema_type: SchemaType | None = Field(default=None, description="Predefined schema type")
    custom_schema: dict[str, Any] | None = Field(default=None, description="Custom JSON schema")
    summary: bool = Field(default=False, description="Generate summary using Gemini")
    timeout: int = Field(default=30, ge=5, le=120, description="Timeout in seconds")
    max_output_chars: int = Field(default=12_000, ge=256, le=100_000)
    max_total_output_chars: int = Field(default=100_000, ge=1_024, le=500_000)
    max_links: int = Field(default=100, ge=0, le=1_000)
    include_chunks: bool = False
    chunk_size: int = Field(default=1_000, ge=256, le=8_000)
    max_chunks: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def validate_extraction_schema(self) -> Self:
        if not self.extract_structured:
            return self
        if self.schema_type is None and self.custom_schema is None:
            message = "extract_structured requires schema_type or custom_schema"
            raise ValueError(message)
        if self.schema_type == SchemaType.CUSTOM and self.custom_schema is None:
            message = "custom schema_type requires custom_schema"
            raise ValueError(message)
        if self.schema_type not in {None, SchemaType.CUSTOM} and self.custom_schema is not None:
            message = "custom_schema requires schema_type=custom"
            raise ValueError(message)
        return self


class CrawlResultItem(BaseModel):
    """Single crawl result."""

    model_config = ConfigDict(extra="forbid")

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
    content_truncated: bool = False
    chunks: list["CrawlChunk"] = Field(default_factory=list)


class CrawlChunk(BaseModel):
    """Bounded content chunk returned when explicitly requested."""

    model_config = ConfigDict(extra="forbid")

    text: str
    index: int
    headers: str
    char_count: int
    word_count: int


class CrawlResponse(BaseModel):
    """Response from crawl operation."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500, description="Search query")
    max_results: int = Field(default=10, ge=1, le=20, description="Max results")
    include_answer: bool = Field(default=True, description="Include AI answer")


class SearchResultItem(BaseModel):
    """Single search result."""

    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    content: str
    score: float
    published_date: str | None = None


class SearchResponse(BaseModel):
    """Response from search operation."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    query: str
    answer: str | None = None
    results: list[SearchResultItem]
    total_results: int


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    model_config = ConfigDict(extra="forbid")

    remaining_minute: int
    remaining_hour: int
