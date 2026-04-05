"""Persistence layer for search ingestion and retrieval."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from app.features.search.constants import (
    DISKANN_QUERY_RESCORE,
    DISKANN_QUERY_SEARCH_LIST_SIZE,
    TRIGRAM_SIMILARITY_THRESHOLD,
)
from app.features.search.fusion import RankedResultRow
from app.features.search.model import SearchChunk, SearchDocument
from app.features.search.rag import SearchChunkRecord

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession


class SearchRepository:
    """Database operations for the search feature."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_document_by_content_hash(self, content_hash: str) -> SearchDocument | None:
        statement = select(SearchDocument).where(SearchDocument.content_hash == content_hash)
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_document_by_id(self, document_id: str) -> SearchDocument | None:
        statement = select(SearchDocument).where(SearchDocument.id == UUID(document_id))
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def create_document(
        self,
        *,
        title: str,
        source_uri: str | None,
        content_hash: str,
        doc_metadata: dict[str, Any],
    ) -> SearchDocument:
        document = SearchDocument(
            title=title,
            source_uri=source_uri,
            content_hash=content_hash,
            doc_metadata=doc_metadata,
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def upsert_chunks(self, rows: list[dict[str, Any]]) -> None:
        """Bulk upsert chunk rows using the document/chunk unique key."""
        if not rows:
            return

        statement = insert(SearchChunk).values(rows)
        statement = statement.on_conflict_do_update(
            constraint="uq_search_chunks_document_chunk_index",
            set_={
                "content": statement.excluded.content,
                "embedding": statement.excluded.embedding,
                "chunk_metadata": statement.excluded.chunk_metadata,
                "updated_at": statement.excluded.updated_at,
            },
        )
        await self.session.execute(statement)

    async def analyze_chunks(self) -> None:
        await self.session.execute(text("ANALYZE search_chunks"))

    async def bm25_search(
        self,
        *,
        query: str,
        candidate_limit: int,
        metadata_filter: dict[str, Any],
    ) -> list[RankedResultRow]:
        statement, filter_params = _build_bm25_statement(metadata_filter)
        params = {
            "query": query,
            "candidate_limit": candidate_limit,
            **filter_params,
        }
        result = await self.session.execute(statement, params)
        return _rank_rows(result.mappings().all())

    async def vector_search(
        self,
        *,
        embedding: list[float],
        candidate_limit: int,
        metadata_filter: dict[str, Any],
    ) -> list[RankedResultRow]:
        statement, filter_params = _build_vector_statement(metadata_filter)
        vector_literal = _vector_literal(embedding)
        await self.session.execute(
            text(f"SET LOCAL diskann.query_search_list_size = {DISKANN_QUERY_SEARCH_LIST_SIZE}")
        )
        await self.session.execute(
            text(f"SET LOCAL diskann.query_rescore = {DISKANN_QUERY_RESCORE}")
        )
        params = {
            "embedding": vector_literal,
            "candidate_limit": candidate_limit,
            **filter_params,
        }
        result = await self.session.execute(statement, params)
        return _rank_rows(result.mappings().all())

    async def trigram_search(
        self,
        *,
        query: str,
        candidate_limit: int,
        metadata_filter: dict[str, Any],
    ) -> list[RankedResultRow]:
        statement, filter_params = _build_trigram_statement(metadata_filter)
        params = {
            "query": query,
            "candidate_limit": candidate_limit,
            "similarity_threshold": TRIGRAM_SIMILARITY_THRESHOLD,
            **filter_params,
        }
        result = await self.session.execute(statement, params)
        return _rank_rows(result.mappings().all())

    async def fetch_chunks_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, SearchChunkRecord]:
        if not chunk_ids:
            return {}

        statement = text(
            """
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                d.title AS title,
                c.content AS content,
                c.chunk_index AS chunk_index,
                c.chunk_metadata AS chunk_metadata
            FROM search_chunks AS c
            JOIN search_documents AS d
              ON d.id = c.document_id
            WHERE c.id = ANY(CAST(:chunk_ids AS uuid[]))
            """
        )
        result = await self.session.execute(statement, {"chunk_ids": list(chunk_ids)})
        return {
            str(row["chunk_id"]): SearchChunkRecord(
                document_id=str(row["document_id"]),
                title=str(row["title"]),
                content=str(row["content"]),
                chunk_index=int(row["chunk_index"]),
                chunk_metadata=dict(row["chunk_metadata"] or {}),
            )
            for row in result.mappings().all()
        }


def _build_bm25_statement(metadata_filter: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if metadata_filter:
        return (
            text(
                """
                SELECT
                    c.id::text AS chunk_id,
                    (-1 * (c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx'))) AS score
                FROM search_chunks AS c
                WHERE (c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')) < 0
                  AND c.chunk_metadata @> CAST(:metadata_filter AS jsonb)
                ORDER BY (c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')) ASC
                LIMIT :candidate_limit
                """
            ),
            {"metadata_filter": json.dumps(metadata_filter)},
        )
    return (
        text(
            """
            SELECT
                c.id::text AS chunk_id,
                (-1 * (c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx'))) AS score
            FROM search_chunks AS c
            WHERE (c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')) < 0
            ORDER BY (c.content <@> to_bm25query(:query, 'search_chunks_bm25_idx')) ASC
            LIMIT :candidate_limit
            """
        ),
        {},
    )


def _build_vector_statement(metadata_filter: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if metadata_filter:
        return (
            text(
                """
                SELECT
                    c.id::text AS chunk_id,
                    (1 - (c.embedding <=> CAST(:embedding AS vector))) AS score
                FROM search_chunks AS c
                WHERE c.embedding IS NOT NULL
                  AND c.chunk_metadata @> CAST(:metadata_filter AS jsonb)
                ORDER BY c.embedding <=> CAST(:embedding AS vector)
                LIMIT :candidate_limit
                """
            ),
            {"metadata_filter": json.dumps(metadata_filter)},
        )
    return (
        text(
            """
            SELECT
                c.id::text AS chunk_id,
                (1 - (c.embedding <=> CAST(:embedding AS vector))) AS score
            FROM search_chunks AS c
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> CAST(:embedding AS vector)
            LIMIT :candidate_limit
            """
        ),
        {},
    )


def _build_trigram_statement(metadata_filter: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if metadata_filter:
        return (
            text(
                """
                SELECT
                    c.id::text AS chunk_id,
                    similarity(c.content, :query) AS score
                FROM search_chunks AS c
                WHERE c.content % :query
                  AND similarity(c.content, :query) >= :similarity_threshold
                  AND c.chunk_metadata @> CAST(:metadata_filter AS jsonb)
                ORDER BY similarity(c.content, :query) DESC
                LIMIT :candidate_limit
                """
            ),
            {"metadata_filter": json.dumps(metadata_filter)},
        )
    return (
        text(
            """
            SELECT
                c.id::text AS chunk_id,
                similarity(c.content, :query) AS score
            FROM search_chunks AS c
            WHERE c.content % :query
              AND similarity(c.content, :query) >= :similarity_threshold
            ORDER BY similarity(c.content, :query) DESC
            LIMIT :candidate_limit
            """
        ),
        {},
    )


def _rank_rows(rows: Sequence[dict[str, Any]]) -> list[RankedResultRow]:
    ranked_rows: list[RankedResultRow] = []
    for rank, row in enumerate(rows, start=1):
        ranked_rows.append(
            RankedResultRow(
                chunk_id=str(row["chunk_id"]),
                score=float(row["score"]),
                rank=rank,
            )
        )
    return ranked_rows


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


def build_chunk_rows(
    *,
    document_id: str,
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create row payloads for repository upsert calls."""
    now = datetime.now(UTC)
    return [
        {
            "id": UUID(str(chunk["id"])),
            "document_id": UUID(document_id),
            "chunk_index": int(chunk["chunk_index"]),
            "content": str(chunk["content"]),
            "embedding": chunk["embedding"],
            "chunk_metadata": dict(chunk["chunk_metadata"]),
            "created_at": now,
            "updated_at": now,
        }
        for chunk in chunks
    ]
