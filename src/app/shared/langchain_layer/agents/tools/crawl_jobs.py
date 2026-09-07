"""Durable crawler-job tools for LLM agents.

Tavily remains the immediate web tool. These six tools expose the durable
Crawl4AI contract without returning raw page dumps by default:

- ``crawl_start`` creates a Redis-backed job and enqueues the Celery task.
- ``crawl_status`` returns a bounded job summary.
- ``crawl_get_page`` returns paginated page summaries.
- ``crawl_get_chunk`` returns paginated chunk texts.
- ``crawl_search_chunks`` returns bounded chunk matches.
- ``crawl_cancel`` requests cooperative cancellation.

Tools are explicitly opt-in via :func:`get_crawl_job_tools` and are NOT part
of the default registry, preserving the Tavily-only default for LLM web
access. Tool-created jobs live under the ``agent`` owner namespace; the REST
API scopes jobs by client identity, so agent jobs are isolated by design.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, override
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

MAX_TOOL_CHARS = 4_000

EnqueueJob = Callable[[str, dict[str, Any]], Awaitable[None]]


def _bound(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 24)] + "\n[truncated for agent]"


class _StoreMixin:
    """Owner-scoped store access shared by durable-job tools.

    Pydantic only honors ``PrivateAttr`` declared directly on the model
    class, so each concrete tool declares its own ``_job_store``/``_owner``;
    this mixin only holds the shared accessors.
    """

    @property
    def store_owner(self) -> str:
        owner: Any = getattr(self, "_owner", "agent")
        return owner if isinstance(owner, str) else "agent"

    def _require_store(self) -> Any:
        store: Any = getattr(self, "_job_store", None)
        if store is None:
            message = (
                "Crawl job store is not configured. "
                "Create tools via get_crawl_job_tools(job_store, ...) or use the REST API."
            )
            raise RuntimeError(message)
        return store


class CrawlStartInput(BaseModel):
    """Input schema for starting a durable crawl."""

    url: str = Field(min_length=1, max_length=4096)
    max_depth: int = Field(default=1, ge=1, le=5)
    max_pages: int = Field(default=10, ge=1, le=50)


class CrawlStatusInput(BaseModel):
    """Input schema for job status."""

    crawl_id: str = Field(min_length=1, max_length=128)


class CrawlGetPageInput(BaseModel):
    """Input schema for paginated page retrieval."""

    crawl_id: str = Field(min_length=1, max_length=128)
    cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=5, ge=1, le=10)


class CrawlGetChunkInput(BaseModel):
    """Input schema for paginated chunk retrieval."""

    crawl_id: str = Field(min_length=1, max_length=128)
    cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=20)


class CrawlSearchChunksInput(BaseModel):
    """Input schema for bounded chunk search."""

    crawl_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=20)


class CrawlCancelInput(BaseModel):
    """Input schema for cancellation."""

    crawl_id: str = Field(min_length=1, max_length=128)


class CrawlStartTool(_StoreMixin, BaseTool):
    """Create a durable crawl job; never runs Crawl4AI inline."""

    name: str = "crawl_start"
    description: str = (
        "Start a durable web crawl. Returns a crawl_id immediately; "
        "execution happens in a Celery worker. Poll crawl_status, then "
        "retrieve pages/chunks. Raw content is never returned here."
    )
    args_schema: type[CrawlStartInput] = CrawlStartInput  # type: ignore[assignment]

    _job_store: Any = PrivateAttr(default=None)
    _owner: str = PrivateAttr(default="agent")
    _enqueue: EnqueueJob | None = PrivateAttr(default=None)

    def __init__(
        self, job_store: Any = None, enqueue_job: EnqueueJob | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._job_store = job_store
        self._enqueue = enqueue_job

    @override
    def _run(self, *args: Any, **kwargs: Any) -> str:
        return asyncio.run(self._arun(*args, **kwargs))

    @override
    async def _arun(self, url: str, max_depth: int = 1, max_pages: int = 10) -> str:
        store = self._require_store()
        crawl_id = uuid4().hex
        payload = {"url": url, "max_depth": max_depth, "max_pages": max_pages}
        job, _ = await store.create(
            crawl_id=crawl_id, owner=self.store_owner, request_payload=payload
        )
        if self._enqueue is None:
            return _bound(
                f"crawl_id={job.crawl_id} status={job.status.value} "
                "(enqueue not configured; job is queued in Redis only)"
            )
        await self._enqueue(job.crawl_id, payload)
        return _bound(f"crawl_id={job.crawl_id} status={job.status.value} url={url}")


class CrawlStatusTool(_StoreMixin, BaseTool):
    """Return a bounded durable-job summary."""

    name: str = "crawl_status"
    description: str = "Return status, page counts, truncation flag, and errors for a crawl_id."
    args_schema: type[CrawlStatusInput] = CrawlStatusInput  # type: ignore[assignment]

    _job_store: Any = PrivateAttr(default=None)
    _owner: str = PrivateAttr(default="agent")

    def __init__(self, job_store: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._job_store = job_store

    @override
    def _run(self, *args: Any, **kwargs: Any) -> str:
        return asyncio.run(self._arun(*args, **kwargs))

    @override
    async def _arun(self, crawl_id: str) -> str:
        store = self._require_store()
        job = await store.get(crawl_id, owner=self.store_owner)
        if job is None:
            return f"crawl_id={crawl_id} status=not_found"
        return _bound(
            f"crawl_id={job.crawl_id} status={job.status.value} "
            f"pages={job.successful_pages}/{job.total_pages} "
            f"failed={job.failed_pages} truncated={job.content_truncated} "
            f"error={job.error_message or 'none'}"
        )


class CrawlGetPageTool(_StoreMixin, BaseTool):
    """Return paginated page summaries for a completed job."""

    name: str = "crawl_get_page"
    description: str = "Return paginated page metadata, summaries, and cursors for a crawl_id."
    args_schema: type[CrawlGetPageInput] = CrawlGetPageInput  # type: ignore[assignment]

    _job_store: Any = PrivateAttr(default=None)
    _owner: str = PrivateAttr(default="agent")

    def __init__(self, job_store: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._job_store = job_store

    @override
    def _run(self, *args: Any, **kwargs: Any) -> str:
        return asyncio.run(self._arun(*args, **kwargs))

    @override
    async def _arun(self, crawl_id: str, cursor: int = 0, limit: int = 5) -> str:
        store = self._require_store()
        result = await store.pages(crawl_id, owner=self.store_owner, cursor=cursor, limit=limit)
        if result is None:
            return f"crawl_id={crawl_id} pages=pending"
        lines = [f"crawl_id={crawl_id} next_cursor={result.next_cursor}"]
        for page in result.items:
            summary = (page.summary or page.title or page.url or "")[:200]
            lines.append(
                f"- page_id={page.page_id} url={page.url} success={page.success} "
                f"truncated={page.content_truncated} chunks={len(page.chunks)} summary={summary}"
            )
        return _bound("\n".join(lines))


class CrawlGetChunkTool(_StoreMixin, BaseTool):
    """Return paginated chunk texts for a completed job."""

    name: str = "crawl_get_chunk"
    description: str = "Return paginated chunk texts with page identity for a crawl_id."
    args_schema: type[CrawlGetChunkInput] = CrawlGetChunkInput  # type: ignore[assignment]

    _job_store: Any = PrivateAttr(default=None)
    _owner: str = PrivateAttr(default="agent")

    def __init__(self, job_store: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._job_store = job_store

    @override
    def _run(self, *args: Any, **kwargs: Any) -> str:
        return asyncio.run(self._arun(*args, **kwargs))

    @override
    async def _arun(self, crawl_id: str, cursor: int = 0, limit: int = 10) -> str:
        store = self._require_store()
        result = await store.chunks(crawl_id, owner=self.store_owner, cursor=cursor, limit=limit)
        if result is None:
            return f"crawl_id={crawl_id} chunks=pending"
        items, next_cursor = result
        lines = [f"crawl_id={crawl_id} next_cursor={next_cursor}"]
        lines.extend(
            f"- page={item.page_url} headers={item.chunk.headers} text={item.chunk.text[:300]}"
            for item in items
        )
        return _bound("\n".join(lines))


class CrawlSearchChunksTool(_StoreMixin, BaseTool):
    """Bounded text search over persisted crawl chunks."""

    name: str = "crawl_search_chunks"
    description: str = "Search persisted chunks for a crawl_id with a bounded match list."
    args_schema: type[CrawlSearchChunksInput] = CrawlSearchChunksInput  # type: ignore[assignment]

    _job_store: Any = PrivateAttr(default=None)
    _owner: str = PrivateAttr(default="agent")

    def __init__(self, job_store: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._job_store = job_store

    @override
    def _run(self, *args: Any, **kwargs: Any) -> str:
        return asyncio.run(self._arun(*args, **kwargs))

    @override
    async def _arun(self, crawl_id: str, query: str, limit: int = 10) -> str:
        store = self._require_store()
        result = await store.search_chunks(
            crawl_id, owner=self.store_owner, query=query, limit=limit
        )
        if result is None:
            return f"crawl_id={crawl_id} search=pending"
        lines = [f"crawl_id={crawl_id} query={query} matches={len(result.items)}"]
        lines.extend(
            f"- page={item.page_url} text={item.chunk.text[:300]}" for item in result.items
        )
        return _bound("\n".join(lines))


class CrawlCancelTool(_StoreMixin, BaseTool):
    """Request cooperative cancellation of a durable crawl job."""

    name: str = "crawl_cancel"
    description: str = "Request cancellation of a queued or running crawl_id."
    args_schema: type[CrawlCancelInput] = CrawlCancelInput  # type: ignore[assignment]

    _job_store: Any = PrivateAttr(default=None)
    _owner: str = PrivateAttr(default="agent")

    def __init__(self, job_store: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._job_store = job_store

    @override
    def _run(self, *args: Any, **kwargs: Any) -> str:
        return asyncio.run(self._arun(*args, **kwargs))

    @override
    async def _arun(self, crawl_id: str) -> str:
        store = self._require_store()
        job = await store.request_cancel(crawl_id, owner=self.store_owner)
        if job is None:
            return f"crawl_id={crawl_id} status=not_found"
        return _bound(f"crawl_id={job.crawl_id} status={job.status.value}")


def get_crawl_job_tools(
    job_store: Any = None,
    enqueue_job: EnqueueJob | None = None,
) -> list[BaseTool]:
    """Build the six durable-job tools with explicit dependencies.

    Not registered by default: import and call this where an agent needs
    durable crawling. Tavily remains the default immediate web tool.
    """
    return [
        CrawlStartTool(job_store=job_store, enqueue_job=enqueue_job),
        CrawlStatusTool(job_store=job_store),
        CrawlGetPageTool(job_store=job_store),
        CrawlGetChunkTool(job_store=job_store),
        CrawlSearchChunksTool(job_store=job_store),
        CrawlCancelTool(job_store=job_store),
    ]
