"""Persistence and retrieval operations for unified documents/chunks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from returns.result import Failure, Success
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.features.search.constants import (
    DISKANN_QUERY_RESCORE,
    DISKANN_QUERY_SEARCH_LIST_SIZE,
    TRIGRAM_SIMILARITY_THRESHOLD,
)
from app.shared.result import ConflictAppError, InfrastructureAppError, NotFoundAppError
from app.utils import ErrorCode

from .model import UnifiedChunk, UnifiedDocument

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.shared.result import AppResult


_FILTER_SQL = """
  AND (:document_ids = '{}' OR c.document_id = ANY(CAST(:document_ids AS uuid[])))
  AND (:document_kind IS NULL OR c.chunk_kind = :document_kind)
  AND (:jurisdiction IS NULL OR c.metadata_->>'jurisdiction' = :jurisdiction)
  AND (:contract_type IS NULL OR c.metadata_->>'contract_type' = :contract_type)
  AND (:clause_type IS NULL OR c.clause_type = :clause_type)
  AND (:require_graphiti_verified IS FALSE OR c.graphiti_verified IS TRUE)
  AND (:metadata_filter = '{}' OR c.metadata_ @> CAST(:metadata_filter AS jsonb))
  AND (:parties_filter = '[]' OR c.metadata_->'parties' @> CAST(:parties_filter AS jsonb))
"""


class DocumentRepository:
    """Repository for unified document lifecycle and retrieval."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_document_by_user_hash(
        self,
        *,
        user_id: str,
        content_hash: str,
    ) -> AppResult[UnifiedDocument | None]:
        try:
            statement = select(UnifiedDocument).where(
                UnifiedDocument.user_id == user_id,
                UnifiedDocument.content_hash == content_hash,
            )
            result = await self.session.execute(statement)
            doc = result.scalar_one_or_none()
            if doc is None:
                return Failure(
                    NotFoundAppError(
                        code=ErrorCode.DOCUMENT_NOT_FOUND,
                        message="Document not found for the given user and content hash",
                        details={"user_id": user_id, "content_hash": content_hash},
                        source="document_repository",
                    )
                )
            return Success(doc)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching document by user hash",
                    details={"user_id": user_id, "content_hash": content_hash, "error": str(exc)},
                    source="document_repository",
                )
            )

    async def get_document_by_id(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> AppResult[UnifiedDocument | None]:
        try:
            statement = select(UnifiedDocument).where(
                UnifiedDocument.user_id == user_id,
                UnifiedDocument.id == UUID(document_id),
            )
            result = await self.session.execute(statement)
            doc = result.scalar_one_or_none()
            if doc is None:
                return Failure(
                    NotFoundAppError(
                        code=ErrorCode.DOCUMENT_NOT_FOUND,
                        message="Document not found for the given user and document ID",
                        details={"user_id": user_id, "document_id": document_id},
                        source="document_repository",
                    )
                )
            return Success(doc)
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching document by ID",
                    details={"user_id": user_id, "document_id": document_id, "error": str(exc)},
                    source="document_repository",
                )
            )

    async def create_document(
        self,
        *,
        user_id: str,
        title: str,
        source_uri: str | None,
        object_uri: str,
        content_hash: str,
        document_kind: str,
        status: str,
        jurisdiction: str | None,
        contract_type: str | None,
        parties: list[object],
        metadata_: dict[str, object],
    ) -> AppResult[UnifiedDocument]:
        try:
            document = UnifiedDocument(
                user_id=user_id,
                title=title,
                source_uri=source_uri,
                object_uri=object_uri,
                content_hash=content_hash,
                document_kind=document_kind,
                status=status,
                jurisdiction=jurisdiction,
                contract_type=contract_type,
                parties=parties,
                metadata_=metadata_,
            )
            self.session.add(document)
            await self.session.flush()
            return Success(document)
        except IntegrityError as exc:
            return Failure(
                ConflictAppError(
                    code="DOCUMENT_CONFLICT",
                    message="Document creation failed due to a constraint violation",
                    details={"user_id": user_id, "content_hash": content_hash, "error": str(exc)},
                    source="document_repository",
                )
            )
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while creating document",
                    details={"user_id": user_id, "content_hash": content_hash, "error": str(exc)},
                    source="document_repository",
                )
            )

    async def update_document_status(
        self,
        *,
        document_id: str,
        status: str,
        title: str | None = None,
        document_kind: str | None = None,
        jurisdiction: str | None = None,
        contract_type: str | None = None,
        parties: list[object] | None = None,
        metadata_: dict[str, object] | None = None,
    ) -> None:
        statement = text(
            """
            UPDATE documents
            SET
                status = :status,
                title = COALESCE(:title, title),
                document_kind = COALESCE(:document_kind, document_kind),
                jurisdiction = COALESCE(:jurisdiction, jurisdiction),
                contract_type = COALESCE(:contract_type, contract_type),
                parties = COALESCE(CAST(:parties AS jsonb), parties),
                metadata_ = COALESCE(CAST(:metadata_ AS jsonb), metadata_),
                updated_at = :updated_at
            WHERE id = :document_id::uuid
            """
        )
        await self.session.execute(
            statement,
            {
                "document_id": document_id,
                "status": status,
                "title": title,
                "document_kind": document_kind,
                "jurisdiction": jurisdiction,
                "contract_type": contract_type,
                "parties": json.dumps(parties) if parties is not None else None,
                "metadata_": json.dumps(metadata_) if metadata_ is not None else None,
                "updated_at": datetime.now(UTC),
            },
        )

    async def upsert_chunks(self, rows: list[dict[str, Any]]) -> AppResult[None]:
        if not rows:
            return Success(None)
        try:
            statement = insert(UnifiedChunk).values(rows)
            statement = statement.on_conflict_do_update(
                constraint="uq_chunks_document_chunk_index",
                set_={
                    "chunk_kind": statement.excluded.chunk_kind,
                    "content": statement.excluded.content,
                    "preamble": statement.excluded.preamble,
                    "clause_type": statement.excluded.clause_type,
                    "page_no": statement.excluded.page_no,
                    "embedding": statement.excluded.embedding,
                    "metadata_": statement.excluded.metadata_,
                    "custom_metadata": statement.excluded.custom_metadata,
                    "quality_warnings": statement.excluded.quality_warnings,
                    "graphiti_episode_id": statement.excluded.graphiti_episode_id,
                    "graphiti_verified": statement.excluded.graphiti_verified,
                },
            )
            await self.session.execute(statement)
            return Success(None)
        except IntegrityError as exc:
            return Failure(
                ConflictAppError(
                    code="CHUNK_CONFLICT",
                    message="Chunk upsert failed due to a constraint violation",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while upserting chunks",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )

    async def analyze_chunks(self) -> None:
        await self.session.execute(text("ANALYZE chunks"))

    async def fetch_status(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> AppResult[dict[str, Any] | None]:
        try:
            statement = text(
                """
                SELECT
                    d.id::text AS document_id,
                    d.status,
                    d.object_uri,
                    d.title,
                    d.document_kind,
                    COUNT(c.id)::int AS chunk_count,
                    COUNT(*) FILTER (WHERE c.graphiti_verified)::int AS verified_chunk_count,
                    COALESCE(jsonb_agg(c.quality_warnings) FILTER (WHERE c.id IS NOT NULL), '[]'::jsonb) AS warnings
                FROM documents AS d
                LEFT JOIN chunks AS c
                  ON c.document_id = d.id
                WHERE d.user_id = :user_id AND d.id = :document_id::uuid
                GROUP BY d.id, d.status, d.object_uri, d.title, d.document_kind
                """
            )
            result = await self.session.execute(
                statement, {"user_id": user_id, "document_id": document_id}
            )
            row = result.mappings().one_or_none()
            if row is None:
                return Failure(
                    NotFoundAppError(
                        code=ErrorCode.STATUS_NOT_FOUND,
                        message="Status not found for the given document",
                        details={"user_id": user_id, "document_id": document_id},
                        source="document_repository",
                    )
                )
            return Success(dict(row))
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while fetching document status",
                    details={"user_id": user_id, "document_id": document_id, "error": str(exc)},
                    source="document_repository",
                )
            )

    async def bm25_search(
        self,
        *,
        user_id: str,
        query: str,
        candidate_limit: int,
        filter_params: dict[str, Any],
    ) -> AppResult[list[dict[str, Any]]]:
        try:
            statement = text(
                """
                SELECT
                    c.id::text AS chunk_id,
                    (-1 * (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx'))) AS score
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.user_id = :user_id
                  AND (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')) < 0
                """
                + _FILTER_SQL
                + """
                ORDER BY (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')) ASC
                LIMIT :candidate_limit
                """
            )
            result = await self.session.execute(
                statement,
                {
                    "user_id": user_id,
                    "query": query,
                    "candidate_limit": candidate_limit,
                    **filter_params,
                },
            )
            return Success([dict(row) for row in result.mappings().all()])
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while performing BM25 search",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )

    async def vector_search(
        self,
        *,
        user_id: str,
        embedding: list[float],
        candidate_limit: int,
        filter_params: dict[str, Any],
    ) -> AppResult[list[dict[str, Any]]]:
        try:
            statement = text(
                """
                SELECT
                    c.id::text AS chunk_id,
                    (1 - (c.embedding <=> CAST(:embedding AS vector))) AS score
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.user_id = :user_id
                  AND c.embedding IS NOT NULL
                """
                + _FILTER_SQL
                + """
                ORDER BY c.embedding <=> CAST(:embedding AS vector)
                LIMIT :candidate_limit
                """
            )
            await self.session.execute(
                text(f"SET LOCAL diskann.query_search_list_size = {DISKANN_QUERY_SEARCH_LIST_SIZE}")
            )
            await self.session.execute(
                text(f"SET LOCAL diskann.query_rescore = {DISKANN_QUERY_RESCORE}")
            )
            result = await self.session.execute(
                statement,
                {
                    "user_id": user_id,
                    "embedding": _vector_literal(embedding),
                    "candidate_limit": candidate_limit,
                    **filter_params,
                },
            )
            return Success([dict(row) for row in result.mappings().all()])
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while performing vector search",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )

    async def trigram_search(
        self,
        *,
        user_id: str,
        query: str,
        candidate_limit: int,
        filter_params: dict[str, Any],
    ) -> AppResult[list[dict[str, Any]]]:
        try:
            statement = text(
                """
                SELECT
                    c.id::text AS chunk_id,
                    similarity(c.search_text, :query) AS score
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.user_id = :user_id
                  AND c.search_text % :query
                  AND similarity(c.search_text, :query) >= :similarity_threshold
                """
                + _FILTER_SQL
                + """
                ORDER BY similarity(c.search_text, :query) DESC
                LIMIT :candidate_limit
                """
            )
            result = await self.session.execute(
                statement,
                {
                    "user_id": user_id,
                    "query": query,
                    "candidate_limit": candidate_limit,
                    "similarity_threshold": TRIGRAM_SIMILARITY_THRESHOLD,
                    **filter_params,
                },
            )
            return Success([dict(row) for row in result.mappings().all()])
        except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while performing trigram search",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )

    async def fetch_chunks_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        statement = text(
            """
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                d.title AS title,
                c.content AS content,
                c.chunk_index,
                c.chunk_kind,
                c.clause_type,
                c.metadata_ AS chunk_metadata,
                c.quality_warnings,
                c.graphiti_verified
            FROM chunks AS c
            JOIN documents AS d ON d.id = c.document_id
            WHERE c.id = ANY(CAST(:chunk_ids AS uuid[]))
            """
        )
        result = await self.session.execute(statement, {"chunk_ids": list(chunk_ids)})
        return {str(row["chunk_id"]): dict(row) for row in result.mappings().all()}

    async def legal_rrf_search(
        self,
        *,
        user_id: str,
        query_text: str,
        query_embedding: list[float],
        limit: int,
        vector_weight: float,
        keyword_weight: float,
        jurisdiction: str | None,
        contract_type: str | None,
        document_ids: Sequence[str] | None,
        chunk_ids: Sequence[str] | None,
        clause_type: str | None,
        require_graphiti_verified: bool,
        bm25_threshold: float | None = None,
        exact_phrase: str | None = None,
    ) -> list[dict[str, Any]]:
        statement = text(
            """
            WITH candidate_chunks AS (
                SELECT
                    c.id,
                    c.content,
                    c.preamble,
                    c.clause_type,
                    c.document_id,
                    c.metadata_,
                    c.custom_metadata,
                    c.embedding,
                    c.search_text,
                    c.quality_warnings,
                    c.graphiti_verified
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.user_id = :user_id
                  AND (:document_ids IS NULL OR c.document_id = ANY(CAST(:document_ids AS uuid[])))
                  AND (:chunk_ids IS NULL OR c.id = ANY(CAST(:chunk_ids AS uuid[])))
                  AND (:jurisdiction IS NULL OR c.metadata_->>'jurisdiction' = :jurisdiction)
                  AND (:contract_type IS NULL OR c.metadata_->>'contract_type' = :contract_type)
                  AND (:clause_type IS NULL OR c.clause_type = :clause_type)
                  AND (:require_graphiti_verified IS FALSE OR c.graphiti_verified IS TRUE)
            ),
            vector_search AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:query_embedding AS vector)) AS rank
                FROM candidate_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:query_embedding AS vector)
                LIMIT 50
            ),
            keyword_search AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        ORDER BY search_text <@> to_bm25query(:query_text, 'chunks_bm25_idx')
                    ) AS rank
                FROM candidate_chunks
                WHERE (
                    :bm25_threshold IS NULL OR
                    search_text <@> to_bm25query(:query_text, 'chunks_bm25_idx') < :bm25_threshold
                )
                ORDER BY search_text <@> to_bm25query(:query_text, 'chunks_bm25_idx')
                LIMIT 50
            ),
            trigram_search AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY similarity(search_text, :query_text) DESC) AS rank
                FROM candidate_chunks
                WHERE search_text % :query_text
                LIMIT 50
            ),
            fused AS (
                SELECT
                    COALESCE(v.id, k.id, t.id) AS id,
                    (:vector_weight * COALESCE(1.0 / (60.0 + v.rank), 0.0)) +
                    (:keyword_weight * COALESCE(1.0 / (60.0 + k.rank), 0.0)) +
                    (0.15 * COALESCE(1.0 / (60.0 + t.rank), 0.0)) AS rrf_score
                FROM vector_search AS v
                FULL OUTER JOIN keyword_search AS k ON v.id = k.id
                FULL OUTER JOIN trigram_search AS t ON COALESCE(v.id, k.id) = t.id
            )
            SELECT
                c.id::text AS chunk_id,
                c.content AS chunk_text,
                c.preamble,
                c.clause_type,
                c.document_id::text AS parent_doc_id,
                c.metadata_,
                c.custom_metadata,
                c.quality_warnings,
                c.graphiti_verified,
                f.rrf_score
            FROM fused AS f
            JOIN chunks AS c ON c.id = f.id
            WHERE (:exact_phrase_like IS NULL OR c.search_text ILIKE :exact_phrase_like)
            ORDER BY f.rrf_score DESC
            LIMIT :limit
            """
        )
        result = await self.session.execute(
            statement,
            {
                "user_id": user_id,
                "query_text": query_text,
                "query_embedding": _vector_literal(query_embedding),
                "limit": limit,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "jurisdiction": jurisdiction,
                "contract_type": contract_type,
                "document_ids": list(document_ids) if document_ids else None,
                "chunk_ids": list(chunk_ids) if chunk_ids else None,
                "clause_type": clause_type,
                "require_graphiti_verified": require_graphiti_verified,
                "bm25_threshold": bm25_threshold,
                "exact_phrase_like": f"%{exact_phrase}%" if exact_phrase else None,
            },
        )
        return [dict(row) for row in result.mappings().all()]


def build_chunk_rows(
    *, document_id: str, user_id: str, chunks: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [{**chunk, "document_id": document_id, "user_id": user_id} for chunk in chunks]


def build_search_filter_params(*, metadata_filter: dict[str, Any]) -> dict[str, Any]:
    document_ids = metadata_filter.get("document_ids") or []
    parties = metadata_filter.get("parties") or []
    return {
        "document_ids": document_ids,
        "document_kind": metadata_filter.get("document_kind"),
        "jurisdiction": metadata_filter.get("jurisdiction"),
        "contract_type": metadata_filter.get("contract_type"),
        "clause_type": metadata_filter.get("clause_type"),
        "require_graphiti_verified": bool(metadata_filter.get("require_graphiti_verified", False)),
        "metadata_filter": json.dumps(metadata_filter.get("metadata_", {})),
        "parties_filter": json.dumps(parties),
    }


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.12f}" for value in embedding) + "]"
