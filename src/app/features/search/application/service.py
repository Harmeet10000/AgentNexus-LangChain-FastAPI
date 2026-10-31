"""Site-search service: ingestion orchestration, hybrid search, RAG assembly."""

from __future__ import annotations

import asyncio
import hashlib
import traceback
from typing import TYPE_CHECKING, Protocol

import orjson
from returns.result import Failure, Success

from app.ai.embeddings import EmbeddingTaskType, embed_text
from app.config import get_settings
from app.events.writer import stamp_outbox_payload_event_id
from app.features.search.adapters.persistence.models import SEARCH_EMBEDDING_DIM
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
from app.features.search.domain.models import (
    TASK_CELERY_ID_CACHE_KEY,
    TASK_OWNER_CACHE_KEY,
    SearchDocumentStatus,
)
from app.kernel.result import log_expected_failure
from app.observability.logging import logger, trace_layer
from app.retrieval.fusion import RankedChunk, reciprocal_rank_fusion
from app.shared.jobs import AggregateRef, is_transaction_scope
from app.shared.jobs.catalog import JOB_CATALOG
from app.shared.jobs.names import SEARCH_INGEST
from app.shared.jobs.payloads import SearchIngestPayload
from app.utils.embedding import stored_width_mismatch, width_mismatch_detail

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence
    from typing import Any

    from redis.asyncio import Redis

    from app.features.search.adapters.persistence.models import SearchDocument
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
        """Create or dedupe a document, then stage the typed ingest job.

        The document row and the outbox event are staged on the request
        session and committed together by ``get_postgres_db`` — a document
        whose job was never staged can never be committed. Only a stored
        document whose pipeline already *completed* answers as a duplicate; a
        pending/failed row is re-enqueued, never treated as done.
        """
        canonical_content = payload.content.strip()
        content_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        if self.job_bus is None:
            # Fail closed BEFORE any write: a document row without a queueable
            # pipeline is a pending orphan, so the enqueue seam being absent is
            # an outage, not a 201.
            return Failure(
                SearchInfrastructureError(
                    message="Task queue unavailable",
                    source="search_service",
                )
            )

        existing_result = await self.repo.get_document_by_content_hash(
            tenant_id=tenant_id, content_hash=content_hash
        )
        if isinstance(existing_result, Failure):
            log_expected_failure(existing_result.failure(), operation="search_ingest")
            return Failure(existing_result.failure())
        existing_document = existing_result.unwrap()
        if existing_document is not None and self._is_completed(existing_document):
            return Success(
                SearchIngestResponse(
                    document_id=str(existing_document.id),
                    task_id=None,
                    status="completed",
                    duplicate=True,
                )
            )

        document: SearchDocument
        if existing_document is not None:
            # Pending/processing/failed row from an earlier attempt: keep it
            # (dedupe identity is (tenant, content_hash)) and re-drive the
            # pipeline instead of reporting a completion that never happened.
            document = existing_document
        else:
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

        document_id = str(document.id)
        # The request session is the transaction scope: the runtime guard
        # checks the one staging capability (async execute) structurally.
        if not is_transaction_scope(self.repo.session):
            await self.repo.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Request session cannot stage transactional jobs",
                    details={"document_id": document_id},
                    source="search_service",
                )
            )
        tx = self.repo.session
        try:
            typed_job = JOB_CATALOG[SEARCH_INGEST].prepare(
                SearchIngestPayload(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    content=canonical_content,
                    content_hash=content_hash,
                    title=payload.title,
                    source_uri=payload.source_uri,
                    doc_metadata=payload.doc_metadata,
                )
            )
            receipt = await self.job_bus.in_transaction(tx).enqueue(
                typed_job,
                aggregate=AggregateRef(aggregate_type="search_document", aggregate_id=document_id),
            )
            # The adapter generates the event id internally, so the payload was
            # staged without it; stamping it into the stored row (same
            # transaction) lets the worker bind its real Celery task id back to
            # the event id the API returned to the client.
            await stamp_outbox_payload_event_id(tx, event_id=receipt.event_id)
        except Exception as exc:  # noqa: BLE001 — staging failure is classified into the typed search error union
            exc.add_note(f"operation=search_ingest, document_id={document_id}")
            # The traceback note plus the structured log are the only durable
            # record: the exception itself is discarded below (EXCEPTION-RULES).
            exc.add_note(traceback.format_exc())
            logger.bind(document_id=document_id, tenant_id=tenant_id).exception(
                "search_ingest_stage_failed"
            )
            # Unwind the flushed document row: the request dependency would
            # otherwise commit it without its job — a permanent pending orphan
            # that the dedupe check would then report as "already ingesting".
            await self.repo.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Could not stage search ingest job",
                    details={"document_id": document_id},
                    source="search_service",
                )
            )

        # The transactional bus is the only enqueue path: its receipt carries
        # the outbox event id, which becomes the pollable task identity.
        task_id = receipt.event_id
        owner_recorded = await self._task_owner_set(
            task_id, tenant_id=tenant_id, document_id=document_id
        )
        if not owner_recorded:
            # Fail closed: without the ownership record the submitting tenant
            # could never authorize GET /ingest/{task_id} for a queued task.
            logger.bind(document_id=document_id, task_id=task_id).warning(
                "search_task_owner_record_failed"
            )
            await self.repo.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Could not record ingest task ownership",
                    details={"document_id": document_id},
                    source="search_service",
                )
            )

        logger.bind(document_id=document_id, task_id=task_id, tenant_id=tenant_id).info(
            "search_ingest_queued"
        )
        return Success(
            SearchIngestResponse(
                document_id=document_id,
                task_id=task_id,
                status="queued",
                duplicate=False,
            )
        )

    @staticmethod
    def _is_completed(document: SearchDocument) -> bool:
        """Whether the stored pipeline state says the chunks are search-ready.

        Unknown stored values coerce to PENDING (the domain enum's contract),
        so a status column written by a foreign code path can only ever cause
        a re-enqueue — never a false "completed".
        """
        return SearchDocumentStatus.coerce(document.status) is SearchDocumentStatus.COMPLETED

    @trace_layer("service")
    async def get_ingest_status(
        self, *, tenant_id: str, task_id: str
    ) -> SearchResult[SearchTaskStatusResponse]:
        """Return ingest status for a task, fail-closed to the caller's tenant.

        Authorization rests on the ownership record written at enqueue
        (``TASK_OWNER_CACHE_KEY``), which exists for every task state — the
        Celery result payload only carries the tenant once a worker has run.
        The payload tenant and the tenant-scoped document row act as
        additional fences; a foreign tenant always gets the same answer as a
        nonexistent task.

        Resolution order: the worker's celery-id mapping (it records the real
        dispatch id against the event id — polling the event id itself would
        report PENDING forever), then the durable document status column, then
        the Celery state (QUEUED while still staged).
        """
        owner = await self._task_owner_get(task_id)
        celery_task_id = await self._task_celery_id_get(task_id)
        state, result_payload, error = await asyncio.to_thread(
            self._celery_snapshot, celery_task_id or task_id
        )

        owner_tenant = str(owner.get("tenant_id", "")) if owner else ""
        payload_tenant = str(result_payload.get("tenant_id", ""))
        # Fail closed: ownership must come from a server-written source, and
        # any source that names a different tenant ends the request here.
        if (owner_tenant and owner_tenant != tenant_id) or (
            payload_tenant and payload_tenant != tenant_id
        ):
            return Failure(
                SearchNotFoundError(
                    message="Ingest task not found",
                    details={"task_id": task_id},
                    source="search_service",
                )
            )
        if not owner_tenant and not payload_tenant:
            return Failure(
                SearchNotFoundError(
                    message="Ingest task not found",
                    details={"task_id": task_id},
                    source="search_service",
                )
            )

        document_id_raw = None
        if owner is not None:
            document_id_raw = owner.get("document_id")
        if not document_id_raw:
            document_id_raw = result_payload.get("document_id")
        document_id = str(document_id_raw) if document_id_raw else None

        result_view: dict[str, object] = dict(result_payload) if result_payload else {}
        document: SearchDocument | None = None
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

        if document is not None and self._is_completed(document):
            # The status column is the durable terminal truth — the broker
            # backend is an eventual cache over it, not the other way around.
            state = "SUCCESS"
        elif state == "PENDING" and owner is not None:
            # The outbox row is staged but no worker has picked it up yet (or
            # it has not recorded its dispatch id); the Celery backend's
            # PENDING is the "never seen this task" state.
            state = "QUEUED"

        return Success(
            SearchTaskStatusResponse(
                task_id=task_id,
                status=state,
                document_id=document_id,
                result=result_view or None,
                error=error,
            )
        )

    async def _task_owner_get(self, task_id: str) -> dict[str, object] | None:
        """The ownership record written at enqueue, or ``None``.

        A dead cache degrades to ``None`` (a named degradation boundary): the
        status lookup then fails closed unless the Celery payload — available
        only after a worker has run — can authorize the caller.
        """
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(TASK_OWNER_CACHE_KEY.format(task_id=task_id))
        except Exception as exc:  # noqa: BLE001 — cache is an optional accelerator
            exc.add_note(f"operation=search_task_owner_get, task_id={task_id}")
            logger.bind(task_id=task_id).warning("search_task_owner_read_failed")
            return None
        if raw is None:
            return None
        raw_text = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)
        parsed = orjson.loads(raw_text)
        return dict(parsed) if isinstance(parsed, dict) else None

    async def _task_celery_id_get(self, task_id: str) -> str | None:
        """The real dispatch id the worker recorded, or ``None`` when queued."""
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(TASK_CELERY_ID_CACHE_KEY.format(task_id=task_id))
        except Exception as exc:  # noqa: BLE001 — cache is an optional accelerator
            exc.add_note(f"operation=search_task_celery_id_get, task_id={task_id}")
            logger.bind(task_id=task_id).warning("search_task_celery_id_read_failed")
            return None
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)

    async def _task_owner_set(self, task_id: str, *, tenant_id: str, document_id: str) -> bool:
        """Record task ownership at enqueue; ``False`` when the cache is down."""
        if self.redis is None:
            return False
        record = orjson.dumps({"tenant_id": tenant_id, "document_id": document_id})
        try:
            await self.redis.set(
                TASK_OWNER_CACHE_KEY.format(task_id=task_id),
                record,
                ex=get_settings().SEARCH_TASK_OWNER_TTL_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to the caller as a typed failure
            exc.add_note(f"operation=search_task_owner_set, task_id={task_id}")
            exc.add_note(traceback.format_exc())
            logger.bind(task_id=task_id).exception("search_task_owner_write_failed")
            return False
        return True

    @staticmethod
    def _celery_snapshot(task_id: str) -> tuple[str, dict[str, object], str | None]:
        """Read the Celery result backend off the event loop.

        ``AsyncResult`` state/result are synchronous backend I/O; running them
        on a worker thread keeps the loop free (P5a guard on the dynamic
        result payload is unchanged).
        """
        from app.connections import (  # noqa: PLC0415 — keeps the web import graph broker-free until the first status read
            celery_app,
        )

        task_result = celery_app.AsyncResult(task_id)
        state = str(task_result.state)
        raw_result = task_result.result
        payload = dict(raw_result) if isinstance(raw_result, dict) else {}
        error = str(raw_result) if state == "FAILURE" and raw_result is not None else None
        return state, payload, error

    @trace_layer("service")
    async def hybrid_search(
        self, *, tenant_id: str, payload: HybridSearchRequest
    ) -> SearchResult[SearchResponse]:
        """Run bm25/vector/trigram, fuse, hydrate, and cache.

        The three branches run **sequentially** on the request-scoped session
        (the same shape as the documents ask path): one ``AsyncSession`` cannot
        serve concurrent queries, and a branch failure rolling the shared
        session back under a still-running branch would corrupt the others.
        """
        width_failure = _width_guard_failure()
        if width_failure is not None:
            return Failure(width_failure)

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
            exc.add_note(traceback.format_exc())
            logger.bind(tenant_id=tenant_id).exception("search_query_embedding_failed")
            return Failure(
                SearchInfrastructureError(
                    message="Query embedding failed",
                    source="search_service",
                )
            )

        branch_results = await self._run_branches(
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

    async def _run_branches(
        self,
        *,
        tenant_id: str,
        query: str,
        query_embedding: list[float],
        candidate_limit: int,
        metadata_filter: dict[str, object],
    ) -> list[tuple[str, SearchResult[list[RankedResultRow]]]]:
        """Run the three retrieval branches one at a time on the shared session.

        Deliberately sequential (mirrors the documents ask path): SQLAlchemy
        AsyncSession rejects concurrent use, so ``asyncio.gather`` over one
        request-scoped repo is a correctness bug, not a perf trade. The
        branches are cheap indexed reads; the fusion cost dominates.
        """
        bm25 = await self.repo.bm25_search(
            tenant_id=tenant_id,
            query=query,
            candidate_limit=candidate_limit,
            metadata_filter=metadata_filter,
        )
        vector = await self.repo.vector_search(
            tenant_id=tenant_id,
            embedding=query_embedding,
            candidate_limit=candidate_limit,
            metadata_filter=metadata_filter,
        )
        trigram = await self.repo.trigram_search(
            tenant_id=tenant_id,
            query=query,
            candidate_limit=candidate_limit,
            metadata_filter=metadata_filter,
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


def _width_guard_failure() -> SearchInfrastructureError | None:
    """The typed failure for a column/setting width drift, or ``None``.

    A column whose width disagrees with the configured producer ranks nothing
    correctly; callers refuse before embedding or querying.
    """
    mismatch = stored_width_mismatch(SEARCH_EMBEDDING_DIM)
    if mismatch is None:
        return None
    stored, expected = mismatch
    return SearchInfrastructureError(
        message=width_mismatch_detail(stored, expected, relation="search_chunks.embedding"),
        details={"stored_dim": stored, "configured_dim": expected},
        source="search_service",
    )


def _resolve_branch_rows(
    branch_results: Sequence[tuple[str, SearchResult[list[RankedResultRow]]]],
) -> SearchResult[list[list[RankedResultRow]]]:
    """Unwrap the branch results in order, naming the first branch that failed.

    A broken branch must fail the whole request — silently fusing the healthy
    remainder would return a 200 byte-identical to a legitimately empty corpus.
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
