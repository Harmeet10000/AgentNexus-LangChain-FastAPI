"""Web crawling API schemas."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, HttpUrl


class CrawlRequest(BaseModel):
    """Web crawl request model."""
    url: HttpUrl
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum crawl depth")
    max_pages: int = Field(default=10, ge=1, le=100, description="Maximum pages to crawl")
    extract_content: bool = Field(default=True, description="Extract and clean content")
    extract_links: bool = Field(default=True, description="Extract links from pages")
    extract_images: bool = Field(default=False, description="Extract images from pages")
    follow_robots_txt: bool = Field(default=True, description="Respect robots.txt")
    wait_time: float = Field(default=1.0, ge=0, le=10, description="Wait time between requests (seconds)")
    user_agent: Optional[str] = Field(default=None, description="Custom user agent")
    save_to_vectors: bool = Field(default=False, description="Save content to vector store")
    namespace: str = Field(default="web_content", description="Vector store namespace")


class SmartCrawlRequest(BaseModel):
    """Smart web crawl request with AI extraction."""
    url: HttpUrl
    extraction_prompt: str = Field(..., description="What to extract from the page")
    css_selector: Optional[str] = Field(default=None, description="CSS selector for specific content")
    wait_for_selector: Optional[str] = Field(default=None, description="Wait for specific element")
    javascript_enabled: bool = Field(default=True, description="Enable JavaScript rendering")
    screenshot: bool = Field(default=False, description="Take screenshot of page")
    extract_structured_data: bool = Field(default=True, description="Extract structured data (JSON-LD, etc.)")


class CrawlStatus(BaseModel):
    """Crawl job status model."""
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    url: str
    pages_crawled: int
    start_time: str
    end_time: Optional[str] = None
    error: Optional[str] = None
    results_summary: Optional[Dict[str, Any]] = None


class CrawlResult(BaseModel):
    """Crawl result model."""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    links: Optional[List[str]] = None
    images: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str


class SmartCrawlResponse(BaseModel):
    """Smart crawl response model."""
    url: str
    extracted_data: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None
    screenshot: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UrlExtractionRequest(BaseModel):
    """URL extraction request model."""
    url: HttpUrl
    css_selectors: Optional[List[str]] = None
    xpath_queries: Optional[List[str]] = None


class UrlExtractionResponse(BaseModel):
    """URL extraction response model."""
    url: str
    extracted_content: Dict[str, Any]
    timestamp: str


class SitemapResponse(BaseModel):
    """Sitemap parsing response model."""
    sitemap_url: str
    total_urls: int
    urls: List[str]


class CrawlJobCancelResponse(BaseModel):
    """Crawl job cancellation response model."""
    message: str
    job_id: str