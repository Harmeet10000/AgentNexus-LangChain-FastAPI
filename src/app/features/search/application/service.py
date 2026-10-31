"""Site-search service: ingestion orchestration, hybrid search, RAG assembly."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, Protocol

import orjson
from returns.result import Failure, Success

from app.ai.embeddings import EmbeddingTaskType, embed_text
from app.config import get_settings
from app.features.search.api.http.dto import (
    RagContextSectionResponse,
    RagSearchResponse,
    SearchIngestResponse,
    SearchResponse,
    SearchResultItem,
    SearchTaskStatusResponse,
)
from app.features.search.application.context_assembly import (
    SearchChunkRecord,
    assemble_rag_context,
)
from app.features.search.domain.errors import (
    SearchInfrastructureError,
    SearchNotFoundError,
)
from app.kernel.result import log_expected_failure
from app.observability.logging import logger, trace_layer
from app.retrieval.fusion import RankedChunk, reciprocal_rank_fusion
from app.shared.jobs.catalog import JOB_CATALOG
from app.shared.jobs.names import SEARCH_INGEST
from app.shared.jobs.payloads import SearchIngestPayload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence
    from typing import Any

    from redis.asyncio import Redis

    from app.features.search.adapters.persistence.postgres.search_repository import (
        SearchRepository,
    )
    from app.features.search.api.http.dto import (
        HybridSearchRequest,
        RagSearchRequest,
        SearchIngestRequest,
    )
    from app.features.search.domain.errors import SearchResult
    from app.retrieval.fusion import RankedResultRow
    from app.shared.jobs import JobBus


class EmbedQueryFn(Protocol):
    """The single embedding call the service needs; test doubles implement it."""

    def __call__(self, text: str, *, task_type: EmbeddingTaskType) -> Awaitable[list[float]]: ...


async def _default_embed_query(text: str, *, task_type: EmbeddingTaskType) -> list[float]:
    return await embed_text(text, task_type=task_type)


class SearchService:
    """Request-scoped orchestration for the site-search feature.

    Everything is tenant-scoped: ``tenant_id`` comes from the authorization
    dependency, never from the request body, and it keys both the repository
    filters and the response cache.
    """

    def __init__(
        self,
        repo: SearchRepository,
        redis: Redis | None,
        job_bus: JobBus | None,
        *,
        embed_query: EmbedQueryFn | None = None,
    ):
        self.repo: SearchRepository = repo
        self.redis: Redis | None = redis
        self.job_bus: JobBus | None = job_bus
        if embed_query is None:
            embed_query = _default_embed_query
        self._embed_query: EmbedQueryFn = embed_query

    @trace_layer("service")
    async def ingest_document(
        self,
        *,
        tenant_id: str,
        payload: SearchIngestRequest,
    ) -> SearchResult[SearchIngestResponse]:
        """Create or dedupe a document, then enqueue the typed ingest job."""
        canonical_content = payload.content.strip()
        content_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()
        existing_result = await self.repo.get_document_by_content_hash(
            tenant_id=tenant_id, content_hash=content_hash
        )
        if isinstance(existing_result, Failure):
            log_expected_failure(existing_result.failure(), operation="search_ingest")
            return Failure(existing_result.failure())
        existing_document = existing_result.unwrap()
        if existing_document is not None:
            return Success(
                SearchIngestResponse(
                    document_id=str(existing_document.id),
                    task_id=None,
                    status="completed",
                    duplicate=True,
                )
            )

        create_result = await self.repo.create_document(
            tenant_id=tenant_id,
            title=payload.title,
            source_uri=payload.source_uri,
            content_hash=content_hash,
            doc_metadata=payload.doc_metadata,
        )
        if isinstance(create_result, Failure):
            log_expected_failure(create_result.failure(), operation="search_ingest")
            return Failure(create_result.failure())
        document = create_result.unwrap()

        if self.job_bus is None:
            # Fail closed: a document row without a queued pipeline is a pending
            # orphan, so the enqueue seam being absent is an outage, not a 201.
            return Failure(
                SearchInfrastructureError(
                    message="Task queue unavailable",
                    details={"document_id": str(document.id)},
                    source="search_service",
                )
            )
        try:
            typed_job = JOB_CATALOG[SEARCH_INGEST].prepare(
                SearchIngestPayload(
                    document_id=str(document.id),
                    tenant_id=tenant_id,
                    content=canonical_content,
                    content_hash=content_hash,
                    title=payload.title,
                    source_uri=payload.source_uri,
                    doc_metadata=payload.doc_metadata,
                )
            )
            receipt = await self.job_bus.enqueue(typed_job)
        except Exception as exc:  # noqa: BLE001 — broker failure is classified into the typed search error union
            exc.add_note(f"operation=search_ingest, document_id={document.id}")
            return Failure(
                SearchInfrastructureError(
                    message="Could not enqueue search ingest job",
                    details={"document_id": str(document.id)},
                    source="search_service",
                )
            )

        logger.bind(
            document_id=str(document.id), task_id=receipt.task_id, tenant_id=tenant_id
        ).info("search_ingest_queued")
        return Success(
            SearchIngestResponse(
                document_id=str(document.id),
                task_id=receipt.task_id,
                status="queued",
                duplicate=False,
            )
        )

    @trace_layer("service")
    async def get_ingest_status(
        self, *, tenant_id: str, task_id: str
    ) -> SearchResult[SearchTaskStatusResponse]:
        """Return ingest status for a task, fail-closed to the caller's tenant.

        The worker's return payload carries the tenant the ingest was created
        for; a foreign tenant gets the same answer as a nonexistent task.
        """
        from app.connections import (  # noqa: PLC0415 — keeps the web import graph broker-free until the first status read
            celery_app,
        )

        task_result = celery_app.AsyncResult(task_id)
        result_payload = task_result.result if isinstance(task_result.result, dict) else {}
        payload_tenant_id = str(result_payload.get("tenant_id", ""))
        document_id_raw = result_payload.get("document_id")
        document_id = str(document_id_raw) if document_id_raw else None

        if payload_tenant_id != tenant_id:
            return Failure(
                SearchNotFoundError(
                    message="Ingest task not found",
                    details={"task_id": task_id},
                    source="search_service",
                )
            )

        error = None
        if task_result.state == "FAILURE":
            error = str(task_result.result)

        result_view: dict[str, object] = dict(result_payload) if result_payload else {}
        if document_id is not None:
            document_result = await self.repo.get_document_by_id(
                tenant_id=tenant_id, document_id=document_id
            )
            if isinstance(document_result, Failure):
                log_expected_failure(document_result.failure(), operation="search_ingest_status")
                return Failure(document_result.failure())
            document = document_result.unwrap()
            if document is not None:
                result_view = {
                    **result_view,
                    "title": document.title,
                    "source_uri": document.source_uri,
                }

        return Success(
            SearchTaskStatusResponse(
                task_id=task_id,
                status=str(task_result.state),
                document_id=document_id,
                result=result_view or None,
                error=error,
            )
        )

    @trace_layer("service")
    async def hybrid_search(
        self, *, tenant_id: str, payload: HybridSearchRequest
    ) -> SearchResult[SearchResponse]:
        """Run bm25/vector/trigram in parallel, fuse, hydrate, and cache."""
        limit = payload.limit or get_settings().SEARCH_DEFAULT_PAGE_SIZE
        candidate_limit = payload.candidate_limit or get_settings().SEARCH_HYBRID_CANDIDATE_LIMIT
        metadata_filter = payload.metadata_filter.chunk_metadata

        cache_key = _build_cache_key("hybrid", tenant_id, payload)
        if not payload.bypass_cache and self.redis is not None:
            cached = await self._cache_get(cache_key)
            if cached is not None:
                # The cache-hit path is still a Result: returning the bare
                # model here would make Success/Failure unstatable for routers.
                return Success(cached)

        try:
            query_embedding = await self._embed_query(
                payload.query, task_type=EmbeddingTaskType.QUERY
            )
        except Exception as exc:  # noqa: BLE001 — embedding-provider failure is classified into the typed search error union
            exc.add_note(f"operation=search_hybrid, tenant_id={tenant_id}")
            return Failure(
                SearchInfrastructureError(
                    message="Query embedding failed",
                    source="search_service",
                )
            )

        branch_results = await self._run_parallel_search(
            tenant_id=tenant_id,
            query=payload.query,
            query_embedding=query_embedding,
            candidate_limit=candidate_limit,
            metadata_filter=metadata_filter,
        )
        rows_result = _resolve_branch_rows(branch_results)
        if isinstance(rows_result, Failure):
            return rows_result
        row_sets = rows_result.unwrap()

        fused = reciprocal_rank_fusion(*row_sets, k=get_settings().SEARCH_RRF_K, limit=limit)
        fetch_result = await self.repo.fetch_chunks_by_ids(
            tenant_id=tenant_id, chunk_ids=[item.chunk_id for item in fused]
        )
        if isinstance(fetch_result, Failure):
            log_expected_failure(fetch_result.failure(), operation="search_hybrid")
            return Failure(fetch_result.failure())
        chunk_lookup = fetch_result.unwrap()

        items = _build_search_items(fused, chunk_lookup)
        response = SearchResponse(items=items, cache_hit=False)
        if not payload.bypass_cache and self.redis is not None:
            await self._cache_set(cache_key, response)
        return Success(response)

    @trace_layer("service")
    async def rag_search(
        self, *, tenant_id: str, payload: RagSearchRequest
    ) -> SearchResult[RagSearchResponse]:
        """Hybrid hits plus token-budgeted, document-ordered context sections."""
        settings = get_settings()
        search_result = await self.hybrid_search(tenant_id=tenant_id, payload=payload)
        if isinstance(search_result, Failure):
            return search_result
        search_response = search_result.unwrap()

        chunk_lookup = {
            item.chunk_id: SearchChunkRecord(
                document_id=item.document_id,
                chunk_index=item.chunk_index,
                content=item.content,
                title=item.title,
                chunk_metadata=item.chunk_metadata,
            )
            for item in search_response.items
        }
        ranked_chunks = [
            RankedChunk(chunk_id=item.chunk_id, score=item.score, rank=item.rank)
            for item in search_response.items
        ]
        context_sections = assemble_rag_context(
            ranked_chunks,
            chunk_lookup,
            max_tokens=payload.max_tokens or settings.SEARCH_RAG_DEFAULT_TOKEN_BUDGET,
        )
        return Success(
            RagSearchResponse(
                items=search_response.items,
                context=[
                    RagContextSectionResponse(
                        document_id=section.document_id,
                        title=section.title,
                        content=section.content,
                        chunk_indices=section.chunk_indices,
                        chunk_metadata=section.chunk_metadata,
                    )
                    for section in context_sections
                ],
                cache_hit=search_response.cache_hit,
            )
        )

    async def _run_parallel_search(
        self,
        *,
        tenant_id: str,
        query: str,
        query_embedding: list[float],
        candidate_limit: int,
        metadata_filter: dict[str, object],
    ) -> list[tuple[str, SearchResult[list[RankedResultRow]]]]:
        bm25, vector, trigram = await asyncio.gather(
            self.repo.bm25_search(
                tenant_id=tenant_id,
                query=query,
                candidate_limit=candidate_limit,
                metadata_filter=metadata_filter,
            ),
            self.repo.vector_search(
                tenant_id=tenant_id,
                embedding=query_embedding,
                candidate_limit=candidate_limit,
                metadata_filter=metadata_filter,
            ),
            self.repo.trigram_search(
                tenant_id=tenant_id,
                query=query,
                candidate_limit=candidate_limit,
                metadata_filter=metadata_filter,
            ),
        )
        return [("bm25", bm25), ("vector", vector), ("trigram", trigram)]

    async def _cache_get(self, cache_key: str) -> SearchResponse | None:
        """Return the cached response, or ``None`` on miss OR backend failure.

        A dead cache must never fail a search; degradation is logged and the
        query proceeds uncached (a named degradation boundary).
        """
        if self.redis is None:
            return None
        try:
            cached = await self.redis.get(cache_key)
        except Exception as exc:  # noqa: BLE001 — cache is an optional accelerator
            exc.add_note(f"operation=search_cache_get, key={cache_key}")
            logger.bind(key=cache_key).warning("search_cache_read_failed")
            return None
        if cached is None:
            return None
        raw = cached.decode("utf-8") if isinstance(cached, bytes | bytearray) else str(cached)
        response = SearchResponse.model_validate_json(raw)
        return response.model_copy(update={"cache_hit": True})

    async def _cache_set(self, cache_key: str, response: SearchResponse) -> None:
        if self.redis is None:
            return
        try:
            ttl = get_settings().SEARCH_CACHE_TTL_SECONDS
            await self.redis.setex(cache_key, ttl, response.model_dump_json())
        except Exception as exc:  # noqa: BLE001 — cache is an optional accelerator
            exc.add_note(f"operation=search_cache_set, key={cache_key}")
            logger.bind(key=cache_key).warning("search_cache_write_failed")


def _resolve_branch_rows(
    branch_results: Sequence[tuple[str, SearchResult[list[RankedResultRow]]]],
) -> SearchResult[list[list[RankedResultRow]]]:
    """Unwrap the parallel branch results, naming the first branch that failed.

    A broken branch must fail the whole request — silently fusing the healthy
    two would return a 200 byte-identical to a legitimately empty corpus.
    """
    row_sets: list[list[RankedResultRow]] = []
    for branch_name, branch_result in branch_results:
        if isinstance(branch_result, Failure):
            failure = branch_result.failure()
            log_expected_failure(failure, operation="search_hybrid")
            return Failure(
                SearchInfrastructureError(
                    message=f"{branch_name} retrieval branch failed: {failure.message}",
                    details={**(failure.details or {}), "branch": branch_name},
                    source="search_service",
                )
            )
        row_sets.append(branch_result.unwrap())
    return Success(row_sets)


def _build_cache_key(kind: str, tenant_id: str, payload: HybridSearchRequest) -> str:
    """Content-address the query and its scope.

    ``tenant_id`` is part of the digest: two tenants issuing the identical
    query must not share a cache entry, or one would read the other's results.
    """
    normalized_query = " ".join(payload.query.lower().split())
    filter_json = orjson.dumps(payload.metadata_filter.chunk_metadata, option=orjson.OPT_SORT_KEYS)
    raw_key = b"|".join(
        [
            kind.encode("utf-8"),
            tenant_id.encode("utf-8"),
            normalized_query.encode("utf-8"),
            filter_json,
            str(payload.limit).encode("utf-8"),
            str(payload.candidate_limit).encode("utf-8"),
        ]
    )
    return "search:" + hashlib.sha256(raw_key).hexdigest()


def _build_search_items(
    fused_results: Sequence[RankedChunk],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[SearchResultItem]:
    items: list[SearchResultItem] = []
    for ranked_chunk in fused_results:
        chunk = chunk_lookup.get(ranked_chunk.chunk_id)
        if chunk is None:
            continue
        items.append(
            SearchResultItem(
                chunk_id=ranked_chunk.chunk_id,
                document_id=str(chunk["document_id"]),
                title=str(chunk["title"]),
                content=str(chunk["content"]),
                chunk_index=int(chunk["chunk_index"]),
                chunk_metadata=dict(chunk["chunk_metadata"]),
                score=ranked_chunk.score,
                rank=ranked_chunk.rank,
            )
        )
    return items
