"""Service layer for hybrid search and ingestion."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING
from uuid import uuid4

import orjson

from app.connections import celery_app, init_db
from app.utils import ServiceUnavailableException, logger

from .chunking import chunk_text
from .constants import (
    ANALYZE_THRESHOLD_CHUNKS,
    DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    INGEST_CHUNK_OVERLAP,
    INGEST_CHUNK_SIZE,
    INGEST_EMBEDDING_BATCH_SIZE,
    RRF_K,
)
from .dto import (
    HybridSearchRequest,
    RagContextSectionResponse,
    RagSearchResponse,
    SearchIngestResponse,
    SearchResponse,
    SearchResultItem,
    SearchTaskStatusResponse,
)
from .embeddings import build_embedding_client
from .fusion import RankedChunk, reciprocal_rank_fusion
from .rag import SearchChunkRecord, assemble_rag_context
from .repository import SearchRepository, build_chunk_rows

if TYPE_CHECKING:
    from collections.abc import Sequence

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from .chunking import TextChunk
    from .dto import RagSearchRequest, SearchIngestRequest
    from .fusion import RankedResultRow


class SearchService:
    """Request-scoped orchestration for the search feature."""

    def __init__(self, repo: SearchRepository, redis: Redis | None = None):
        self.repo = repo
        self.redis = redis

    async def ingest_document(self, payload: SearchIngestRequest) -> SearchIngestResponse:
        """Create or deduplicate a document, then enqueue chunk/embed work."""
        canonical_content = payload.content.strip()
        content_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()
        existing_document = await self.repo.get_document_by_content_hash(content_hash)
        if existing_document is not None:
            return SearchIngestResponse(
                document_id=str(existing_document.id),
                task_id=None,
                status="completed",
                duplicate=True,
            )

        document = await self.repo.create_document(
            title=payload.title,
            source_uri=payload.source_uri,
            content_hash=content_hash,
            doc_metadata=payload.doc_metadata,
        )

        try:
            task = celery_app.send_task(
                "tasks.search_ingest",
                kwargs={
                    "document_id": str(document.id),
                    "content": canonical_content,
                    "content_hash": content_hash,
                    "title": payload.title,
                    "source_uri": payload.source_uri,
                    "doc_metadata": payload.doc_metadata,
                },
            )
        except Exception as exc:
            raise ServiceUnavailableException(
                detail="Task queue unavailable",
                data={"document_id": str(document.id)},
            ) from exc

        logger.bind(document_id=str(document.id), task_id=task.id).info("search_ingest_queued")
        return SearchIngestResponse(
            document_id=str(document.id),
            task_id=task.id,
            status="queued",
            duplicate=False,
        )

    async def get_ingest_status(self, task_id: str) -> SearchTaskStatusResponse:
        """Return ingestion status from Celery and, when possible, document metadata."""
        task_result = celery_app.AsyncResult(task_id)
        result_payload = task_result.result if isinstance(task_result.result, dict) else None
        document_id = None
        if result_payload is not None:
            raw_document_id = result_payload.get("document_id")
            document_id = str(raw_document_id) if raw_document_id else None

        error = None
        if task_result.state == "FAILURE":
            error = str(task_result.result)

        if document_id is not None:
            document = await self.repo.get_document_by_id(document_id)
            if document is not None:
                result_payload = {
                    **(result_payload or {}),
                    "title": document.title,
                    "source_uri": document.source_uri,
                }

        return SearchTaskStatusResponse(
            task_id=task_id,
            status=task_result.state,
            document_id=document_id,
            result=result_payload,
            error=error,
        )

    async def hybrid_search(self, payload: HybridSearchRequest) -> SearchResponse:
        """Run BM25 and vector retrieval, fuse the results, and hydrate ranked chunks."""
        cache_key = _build_cache_key("hybrid", payload)
        if not payload.bypass_cache and self.redis is not None:
            cached_response = await self.redis.get(cache_key)
            if cached_response is not None:
                response = SearchResponse.model_validate_json(cached_response)
                return response.model_copy(update={"cache_hit": True})

        embedding_client = build_embedding_client()
        query_embedding = await embedding_client.aembed_query(
            payload.query,
            task_type="RETRIEVAL_QUERY",
        )

        bm25_results, vector_results, trigram_results = await _run_parallel_search(
            self.repo,
            query=payload.query,
            query_embedding=query_embedding,
            candidate_limit=payload.candidate_limit,
            metadata_filter=payload.metadata_filter.chunk_metadata,
        )

        fused_results = reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            trigram_results,
            k=RRF_K,
            limit=payload.limit,
        )
        chunk_lookup = await self.repo.fetch_chunks_by_ids(
            [item.chunk_id for item in fused_results]
        )
        items = _build_search_items(fused_results, chunk_lookup)
        response = SearchResponse(items=items, cache_hit=False)

        if not payload.bypass_cache and self.redis is not None:
            await self.redis.setex(
                cache_key,
                DEFAULT_SEARCH_CACHE_TTL_SECONDS,
                response.model_dump_json(),
            )

        return response

    async def rag_search(self, payload: RagSearchRequest) -> RagSearchResponse:
        """Return ranked hits plus ordered context sections for RAG use."""
        search_response = await self.hybrid_search(
            HybridSearchRequest(
                query=payload.query,
                limit=payload.limit,
                candidate_limit=payload.candidate_limit,
                metadata_filter=payload.metadata_filter,
                bypass_cache=payload.bypass_cache,
            )
        )
        chunk_lookup: dict[str, SearchChunkRecord] = {
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
            ranked_chunks, chunk_lookup, max_tokens=payload.max_tokens
        )

        return RagSearchResponse(
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


async def process_ingestion_document(
    *,
    session: AsyncSession,
    document_id: str,
    content: str,
) -> dict[str, object]:
    """Chunk, embed, and upsert search chunks for a document."""
    repo = SearchRepository(session)
    embedding_client = build_embedding_client()
    chunks = chunk_text(
        content,
        chunk_size=INGEST_CHUNK_SIZE,
        chunk_overlap=INGEST_CHUNK_OVERLAP,
    )

    if not chunks:
        return {
            "status": "completed",
            "document_id": document_id,
            "chunk_count": 0,
        }

    chunk_payloads: list[dict[str, object]] = []
    for batch in _batched(chunks, INGEST_EMBEDDING_BATCH_SIZE):
        embeddings = await embedding_client.aembed_documents(
            [chunk.content for chunk in batch],
            task_type="RETRIEVAL_DOCUMENT",
        )
        for chunk, embedding in zip(batch, embeddings, strict=True):
            chunk_payloads.append(
                {
                    "id": str(uuid4()),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "embedding": embedding,
                    "chunk_metadata": {"token_count": chunk.token_count},
                }
            )

    await repo.upsert_chunks(
        build_chunk_rows(
            document_id=document_id,
            chunks=chunk_payloads,
        )
    )
    if len(chunk_payloads) > ANALYZE_THRESHOLD_CHUNKS:
        await repo.analyze_chunks()

    return {
        "status": "completed",
        "document_id": document_id,
        "chunk_count": len(chunk_payloads),
    }


async def run_ingestion_task(
    *,
    document_id: str,
    content: str,
) -> dict[str, object]:
    """Create a task-local async DB session and execute search ingestion."""
    engine, session_local = await init_db()
    try:
        async with session_local() as session, session.begin():
            return await process_ingestion_document(
                session=session,
                document_id=document_id,
                content=content,
            )
    finally:
        await engine.dispose()


async def _run_parallel_search(
    repo: SearchRepository,
    *,
    query: str,
    query_embedding: list[float],
    candidate_limit: int,
    metadata_filter: dict[str, object],
) -> tuple[list[RankedResultRow], list[RankedResultRow], list[RankedResultRow]]:
    return await asyncio.gather(
        repo.bm25_search(
            query=query,
            candidate_limit=candidate_limit,
            metadata_filter=metadata_filter,
        ),
        repo.vector_search(
            embedding=query_embedding,
            candidate_limit=candidate_limit,
            metadata_filter=metadata_filter,
        ),
        repo.trigram_search(
            query=query,
            candidate_limit=candidate_limit,
            metadata_filter=metadata_filter,
        ),
    )


def _build_search_items(
    fused_results: Sequence[RankedChunk],
    chunk_lookup: dict[str, SearchChunkRecord],
) -> list[SearchResultItem]:
    items: list[SearchResultItem] = []
    for ranked_chunk in fused_results:
        chunk = chunk_lookup.get(ranked_chunk.chunk_id)
        if chunk is None:
            continue
        items.append(
            SearchResultItem(
                chunk_id=ranked_chunk.chunk_id,
                document_id=chunk.document_id,
                title=chunk.title,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                chunk_metadata=chunk.chunk_metadata,
                score=ranked_chunk.score,
                rank=ranked_chunk.rank,
            )
        )
    return items


def _batched(values: Sequence[TextChunk], batch_size: int) -> list[Sequence[TextChunk]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _build_cache_key(kind: str, payload: HybridSearchRequest) -> str:
    normalized_query = " ".join(payload.query.lower().split())
    filter_json = orjson.dumps(payload.metadata_filter.chunk_metadata, option=orjson.OPT_SORT_KEYS)
    raw_key = b"|".join(
        [
            kind.encode("utf-8"),
            normalized_query.encode("utf-8"),
            filter_json,
            str(payload.limit).encode("utf-8"),
            str(payload.candidate_limit).encode("utf-8"),
        ]
    )
    return "search:" + hashlib.sha256(raw_key).hexdigest()
