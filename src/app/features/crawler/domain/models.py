"""Crawler domain and persisted models."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CrawlResultItem(BaseModel):
    """Single crawl result."""

    model_config = ConfigDict(extra="forbid")

    url: str
    page_id: str | None = None
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
    processing_errors: list[str] = Field(default_factory=list)
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
    token_count: int | None = None
    content_hash: str | None = None
    truncated: bool = False


class CrawlJobStatus(StrEnum):
    """Durable lifecycle states for a crawler job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class CrawlJob(BaseModel):
    """Public durable crawler-job representation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    crawl_id: str
    status: CrawlJobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_pages: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    next_page_cursor: str | None = None
    error_message: str | None = None
    content_truncated: bool = False


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for job records."""
    return datetime.now(UTC)
