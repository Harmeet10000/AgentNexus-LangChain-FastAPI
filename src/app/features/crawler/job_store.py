"""Durable Redis state for asynchronous crawler jobs."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

from app.config import get_settings
from app.utils import logger

from .dto import (
    CrawlJob,
    CrawlJobChunk,
    CrawlJobPageList,
    CrawlJobSearchResponse,
    CrawlJobStatus,
    CrawlResponse,
    utc_now,
)


def _stable_page_id(crawl_id: str, page_url: str | None) -> str:
    """Deterministic page identity so chunks stay joinable to their page.

    Recursive-path results predate page_id assignment; hashing crawl_id:url
    at read time backfills them (and any future missing IDs) without a
    data migration.
    """
    seed = f"{crawl_id}:{page_url or ''}"
    return hashlib.sha256(seed.encode()).hexdigest()[:24]


class CrawlJobStore:
    """Own the Redis representation and lifecycle transitions of crawler jobs."""

    def __init__(self, redis: Redis, *, ttl_seconds: int | None = None) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds or get_settings().CRAWL_JOB_TTL_SECONDS

    @staticmethod
    def _meta_key(crawl_id: str) -> str:
        return f"crawler:job:{crawl_id}:meta"

    @staticmethod
    def _result_key(crawl_id: str) -> str:
        return f"crawler:job:{crawl_id}:result"

    @staticmethod
    def _idempotency_key(owner: str, key: str) -> str:
        return f"crawler:idempotency:{owner}:{key}"

    @staticmethod
    def _cancel_key(crawl_id: str) -> str:
        return f"crawler:job:{crawl_id}:cancel"

    async def create(
        self,
        *,
        crawl_id: str,
        owner: str,
        request_payload: dict[str, object],
        idempotency_key: str | None = None,
    ) -> tuple[CrawlJob, bool]:
        """Create a job, returning an existing job for a repeated idempotency key."""
        if idempotency_key:
            existing_id = await self.redis.get(self._idempotency_key(owner, idempotency_key))
            if existing_id:
                existing = await self.get(str(existing_id), owner=owner)
                if existing is not None:
                    return existing, False

        now = utc_now()
        job = CrawlJob(crawl_id=crawl_id, status=CrawlJobStatus.PENDING, created_at=now)
        payload = {
            **job.model_dump(mode="json"),
            "owner": owner,
            "request": request_payload,
        }
        created = await self.redis.set(
            self._meta_key(crawl_id),
            json.dumps(payload, separators=(",", ":")),
            ex=self.ttl_seconds,
            nx=True,
        )
        if not created:
            existing = await self.get(crawl_id, owner=owner)
            if existing is None:
                message = "Crawler job creation raced with an expired record"
                raise RuntimeError(message)
            return existing, False
        if idempotency_key:
            idempotency_created = await self.redis.set(
                self._idempotency_key(owner, idempotency_key),
                crawl_id,
                ex=self.ttl_seconds,
                nx=True,
            )
            if not idempotency_created:
                winner_id = await self.redis.get(self._idempotency_key(owner, idempotency_key))
                if winner_id:
                    winner = await self.get(str(winner_id), owner=owner)
                    if winner is not None:
                        await self.redis.delete(self._meta_key(crawl_id))
                        return winner, False
        return job, True

    async def get(self, crawl_id: str, *, owner: str) -> CrawlJob | None:
        payload = await self._read_payload(crawl_id)
        if payload is None or payload.get("owner") != owner:
            return None
        return self._public_job(payload)

    @staticmethod
    def _public_job(payload: dict[str, object]) -> CrawlJob:
        fields = set(CrawlJob.model_fields)
        return CrawlJob.model_validate(
            {key: value for key, value in payload.items() if key in fields}
        )

    async def _read_payload(self, crawl_id: str) -> dict[str, object] | None:
        raw = await self.redis.get(self._meta_key(crawl_id))
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            logger.bind(crawl_id=crawl_id).exception("Invalid crawler job metadata", error=str(exc))
            return None
        return payload if isinstance(payload, dict) else None

    async def _update(self, crawl_id: str, **changes: object) -> CrawlJob:
        payload = await self._read_payload(crawl_id)
        if payload is None:
            raise KeyError(crawl_id)
        payload.update(changes)
        await self.redis.set(
            self._meta_key(crawl_id),
            json.dumps(payload, separators=(",", ":")),
            ex=self.ttl_seconds,
        )
        return self._public_job(payload)

    async def mark_running(self, crawl_id: str) -> CrawlJob:
        return await self._update(
            crawl_id, status=CrawlJobStatus.RUNNING, started_at=utc_now().isoformat()
        )

    async def attach_task_id(self, crawl_id: str, task_id: str) -> None:
        await self._update(crawl_id, task_id=task_id)

    async def task_id(self, crawl_id: str, *, owner: str) -> str | None:
        payload = await self._read_payload(crawl_id)
        if payload is None or payload.get("owner") != owner:
            return None
        value = payload.get("task_id")
        return value if isinstance(value, str) else None

    async def mark_failed(self, crawl_id: str, message: str) -> CrawlJob:
        return await self._update(
            crawl_id,
            status=CrawlJobStatus.FAILED,
            completed_at=utc_now().isoformat(),
            error_message=message[:2_000],
        )

    async def mark_cancelled(self, crawl_id: str) -> CrawlJob:
        return await self._update(
            crawl_id,
            status=CrawlJobStatus.CANCELLED,
            completed_at=utc_now().isoformat(),
        )

    async def request_cancel(self, crawl_id: str, *, owner: str) -> CrawlJob | None:
        job: CrawlJob | None = await self.get(crawl_id, owner=owner)
        if job is None or job.status in {
            CrawlJobStatus.COMPLETED,
            CrawlJobStatus.PARTIAL,
            CrawlJobStatus.FAILED,
            CrawlJobStatus.CANCELLED,
        }:
            return job
        await self.redis.set(self._cancel_key(crawl_id), "1", ex=self.ttl_seconds)
        return await self._update(crawl_id, status=CrawlJobStatus.CANCELLING)

    async def is_cancelled(self, crawl_id: str) -> bool:
        return bool(await self.redis.exists(self._cancel_key(crawl_id)))

    async def save_result(self, crawl_id: str, result: CrawlResponse) -> CrawlJob:
        encoded = result.model_dump_json().encode("utf-8")
        if len(encoded) > get_settings().CRAWL_JOB_MAX_RESULT_BYTES:
            return await self.mark_failed(
                crawl_id, "Persisted crawler result exceeded the configured size limit"
            )
        await self.redis.set(self._result_key(crawl_id), encoded, ex=self.ttl_seconds)
        status = CrawlJobStatus.PARTIAL if result.failed_pages else CrawlJobStatus.COMPLETED
        return await self._update(
            crawl_id,
            status=status,
            completed_at=utc_now().isoformat(),
            total_pages=result.total_pages,
            successful_pages=result.successful_pages,
            failed_pages=result.failed_pages,
            content_truncated=result.content_truncated,
            next_page_cursor="0" if result.total_pages else None,
        )

    async def get_result(self, crawl_id: str, *, owner: str) -> CrawlResponse | None:
        if await self.get(crawl_id, owner=owner) is None:
            return None
        raw = await self.redis.get(self._result_key(crawl_id))
        if raw is None:
            return None
        try:
            return CrawlResponse.model_validate_json(raw)
        except (ValueError, TypeError) as exc:
            logger.bind(crawl_id=crawl_id).exception(
                "Invalid persisted crawler result", error=str(exc)
            )
            return None

    async def pages(
        self, crawl_id: str, *, owner: str, cursor: int, limit: int
    ) -> CrawlJobPageList | None:
        result = await self.get_result(crawl_id, owner=owner)
        if result is None:
            return None
        items = result.results[cursor : cursor + limit]
        next_cursor = str(cursor + limit) if cursor + limit < len(result.results) else None
        return CrawlJobPageList(crawl_id=crawl_id, items=items, next_cursor=next_cursor)

    async def chunks(
        self, crawl_id: str, *, owner: str, cursor: int, limit: int
    ) -> tuple[list[CrawlJobChunk], str | None] | None:
        result = await self.get_result(crawl_id, owner=owner)
        if result is None:
            return None
        chunks = [
            CrawlJobChunk(
                crawl_id=crawl_id,
                page_id=page.page_id or _stable_page_id(crawl_id, page.url),
                page_url=page.url,
                chunk=chunk,
            )
            for page in result.results
            for chunk in page.chunks
        ]
        items = chunks[cursor : cursor + limit]
        next_cursor = str(cursor + limit) if cursor + limit < len(chunks) else None
        return items, next_cursor

    async def search_chunks(
        self, crawl_id: str, *, owner: str, query: str, limit: int
    ) -> CrawlJobSearchResponse | None:
        result = await self.get_result(crawl_id, owner=owner)
        if result is None:
            return None
        needle = query.casefold()
        items: list[CrawlJobChunk] = []
        for page in result.results:
            for chunk in page.chunks:
                if needle in chunk.text.casefold():
                    items.append(
                        CrawlJobChunk(
                            crawl_id=crawl_id,
                            page_id=page.page_id or _stable_page_id(crawl_id, page.url),
                            page_url=page.url,
                            chunk=chunk,
                        )
                    )
                    if len(items) >= limit:
                        return CrawlJobSearchResponse(crawl_id=crawl_id, query=query, items=items)
        return CrawlJobSearchResponse(crawl_id=crawl_id, query=query, items=items)
