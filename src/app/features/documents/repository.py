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

from app.shared.result.diagnostics import add_database_error_note
from app.utils import trace_layer
from app.utils.embedding import stored_width_mismatch, width_mismatch_detail

from .constants import (
    DISKANN_QUERY_RESCORE,
    DISKANN_QUERY_SEARCH_LIST_SIZE,
    PHRASE_OVERFETCH_MULTIPLE,
    TRIGRAM_SIMILARITY_THRESHOLD,
)
from .errors import (
    DocumentChunkConflictError,
    DocumentConflictError,
    DocumentDatabaseError,
    DocumentEmbeddingWidthError,
    DocumentNotFoundError,
    DocumentStatusNotFoundError,
)
from .model import CHUNK_EMBEDDING_DIM, UnifiedChunk, UnifiedDocument

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy.dialects.postgresql.dml import Insert
    from sqlalchemy.engine.result import Result
    from sqlalchemy.engine.row import RowMapping
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import TextClause
    from sqlalchemy.sql.selectable import Select

    from .errors import DocumentResult

_FILTER_SQL = """
  AND (
    COALESCE(array_length(CAST(:document_ids AS uuid[]), 1), 0) = 0
    OR c.document_id = ANY(CAST(:document_ids AS uuid[]))
  )
  AND (
    COALESCE(array_length(CAST(:chunk_ids AS uuid[]), 1), 0) = 0
    OR c.id = ANY(CAST(:chunk_ids AS uuid[]))
  )
  AND (
    CAST(:document_kind AS text) IS NULL
    OR c.chunk_kind = CAST(:document_kind AS text)
  )
  AND (
    CAST(:jurisdiction AS text) IS NULL
    OR c.metadata_->>'jurisdiction' = CAST(:jurisdiction AS text)
  )
  AND (
    CAST(:contract_type AS text) IS NULL
    OR c.metadata_->>'contract_type' = CAST(:contract_type AS text)
  )
  AND (
    CAST(:clause_type AS text) IS NULL
    OR c.clause_type = CAST(:clause_type AS text)
  )
  AND (
    CAST(:require_graphiti_verified AS boolean) IS FALSE
    OR c.graphiti_verified IS TRUE
  )
  AND (
    CAST(:metadata_filter AS text) = '{}'
    OR c.metadata_ @> CAST(:metadata_filter AS jsonb)
  )
  AND (
    CAST(:parties_filter AS text) = '[]'
    OR c.metadata_->'parties' @> CAST(:parties_filter AS jsonb)
  )
"""


def _phrase_like_pattern(phrase: str) -> str:
    """Build an escaped `LIKE %phrase%` pattern matching the phrase literally.

    The keyword extension has no phrase syntax, so the leg over-fetches on
    relevance and post-filters with this pattern. The wildcard (`%`, `_`) and
    escape (`\\`) characters are escaped so a phrase containing them matches
    literally rather than as a pattern.
    """
    escaped = phrase.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class DocumentRepository:
    """Repository for unified document lifecycle and retrieval."""

    def __init__(self, session: AsyncSession):
        self.session: AsyncSession = session

    @trace_layer("repository")
    async def get_document_by_user_hash(
        self,
        *,
        user_id: str,
        content_hash: str,
    ) -> DocumentResult[UnifiedDocument | None]:
        try:
            statement: Select[tuple[UnifiedDocument]] = select(UnifiedDocument).where(
                UnifiedDocument.user_id == user_id,
                UnifiedDocument.content_hash == content_hash,
            )
            result: Result[tuple[UnifiedDocument]] = await self.session.execute(statement)
            doc: UnifiedDocument | None = result.scalar_one_or_none()
            if doc is None:
                return Failure(
                    inner_value=DocumentNotFoundError(
                        message="Document not found for the given user and content hash",
                        details={"user_id": user_id, "content_hash": content_hash},
                        source="document_repository",
                    )
                )
            return Success(doc)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while fetching document by user hash",
                    details={"user_id": user_id, "content_hash": content_hash, "error": str(exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def get_document_by_id(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> DocumentResult[UnifiedDocument | None]:
        try:
            statement: Select[tuple[UnifiedDocument]] = select(UnifiedDocument).where(
                UnifiedDocument.user_id == user_id,
                UnifiedDocument.id == UUID(document_id),
            )
            result: Result[tuple[UnifiedDocument]] = await self.session.execute(statement)
            doc: UnifiedDocument | None = result.scalar_one_or_none()
            if doc is None:
                return Failure(
                    inner_value=DocumentNotFoundError(
                        message="Document not found for the given user and document ID",
                        details={"user_id": user_id, "document_id": document_id},
                        source="document_repository",
                    )
                )
            return Success(inner_value=doc)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while fetching document by ID",
                    details={
                        "user_id": user_id,
                        "document_id": document_id,
                        "error": str(object=exc),
                    },
                    source="document_repository",
                )
            )

    @trace_layer("repository")
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
    ) -> DocumentResult[UnifiedDocument]:
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
            self.session.add(instance=document)
            await self.session.flush()
            return Success(inner_value=document)
        except IntegrityError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentConflictError(
                    message="Document creation failed due to a constraint violation",
                    details={"user_id": user_id, "content_hash": content_hash, "error": str(exc)},
                    source="document_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while creating document",
                    details={"user_id": user_id, "content_hash": content_hash, "error": str(exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
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
        structural_tree: dict[str, object] | None = None,
        extraction_incomplete: bool | None = None,
    ) -> DocumentResult[None]:
        try:
            statement: TextClause = text(
                text="""
            UPDATE documents
            SET
                status = :status,
                title = COALESCE(:title, title),
                document_kind = COALESCE(:document_kind, document_kind),
                jurisdiction = COALESCE(:jurisdiction, jurisdiction),
                contract_type = COALESCE(:contract_type, contract_type),
                parties = COALESCE(CAST(:parties AS jsonb), parties),
                metadata_ = COALESCE(CAST(:metadata_ AS jsonb), metadata_),
                structural_tree = COALESCE(CAST(:structural_tree AS jsonb), structural_tree),
                extraction_incomplete = COALESCE(:extraction_incomplete, extraction_incomplete),
                updated_at = :updated_at
            WHERE id = CAST(:document_id AS uuid)
            """
            )
            await self.session.execute(
                statement,
                params={
                    "document_id": document_id,
                    "status": status,
                    "title": title,
                    "document_kind": document_kind,
                    "jurisdiction": jurisdiction,
                    "contract_type": contract_type,
                    "parties": json.dumps(obj=parties) if parties is not None else None,
                    "metadata_": json.dumps(obj=metadata_) if metadata_ is not None else None,
                    "structural_tree": (
                        json.dumps(obj=structural_tree) if structural_tree is not None else None
                    ),
                    "extraction_incomplete": extraction_incomplete,
                    "updated_at": datetime.now(tz=UTC),
                },
            )
            return Success(None)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                DocumentDatabaseError(
                    message="Database error while updating document status",
                    details={"document_id": document_id, "error": str(exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def fetch_structural_trees(
        self,
        *,
        user_id: str,
        document_ids: list[str],
    ) -> DocumentResult[list[dict[str, Any]]]:
        """Load tenant-scoped persisted trees for structural navigation."""
        try:
            statement = text(
                """
                SELECT id::text AS document_id, structural_tree
                FROM documents
                WHERE user_id = :user_id
                  AND structural_tree <> '{}'::jsonb
                  AND (
                    cardinality(CAST(:document_ids AS uuid[])) = 0
                    OR id = ANY(CAST(:document_ids AS uuid[]))
                  )
                ORDER BY id
                """
            )
            result = await self.session.execute(
                statement,
                params={"user_id": user_id, "document_ids": document_ids},
            )
            return Success([dict(row) for row in result.mappings().all()])
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                DocumentDatabaseError(
                    message="Database error while loading document structure",
                    details={"user_id": user_id, "error": str(exc)},
                    source="document_repository",
                )
            )

    @staticmethod
    def _reject_width_mismatch(rows: list[dict[str, Any]]) -> DocumentEmbeddingWidthError | None:
        """Refuse a chunk batch whose vectors are not the width this relation stores.

        Returns the error rather than raising it because the caller returns
         ``DocumentResult`` — the raising half of the same guard lives in
        ``utils/embedding.assert_stored_width_matches_configured`` for the offline
        batch paths that have no ``Result`` to put a failure into.

        Two distinct conditions, in the order they can occur:

        1. **Declared width against configured width.** Tautological in production
           today, because ``UnifiedChunk.embedding`` derives its width from
           ``EMBEDDING_DIMENSION`` when the class body runs. It is *not*
           tautological the moment configuration is reloaded after import, and it
           is the condition the N6 stub test drives — there are zero stored
           vectors, so the only honest way to exercise a stored-width mismatch is
           to move the configured value out from under a column already built.
        2. **Row width against declared width.** This is the one that fires in
           practice: a caller hands vectors from a model of a different shape.
           Refusing here rather than letting psycopg reject the INSERT is what
           turns an opaque driver error deep in a batch into a diagnostic that
           names the relation, both widths, and the remedy.
        """
        stored_dim = CHUNK_EMBEDDING_DIM

        mismatch = stored_width_mismatch(stored_dim)
        if mismatch is not None:
            stored, expected = mismatch
            return DocumentEmbeddingWidthError(
                message=width_mismatch_detail(stored, expected, relation="chunks.embedding"),
                details={"stored_dim": stored, "configured_dim": expected},
                # `retryable` defaults to True on this error, which would be wrong
                # here in a way that costs real money: a Celery task retrying a
                # width disagreement spins until its ceiling against a condition
                # that only a re-embedding run can change.
                source="document_repository",
            )

        for index, row in enumerate(iterable=rows):
            embedding = row.get("embedding")
            if embedding is None:
                continue
            actual = len(embedding)
            if actual != stored_dim:
                return DocumentEmbeddingWidthError(
                    message=(
                        f"chunk at index {index} carries a {actual}-dimensional vector but "
                        f"chunks.embedding stores {stored_dim}; the batch is refused rather "
                        f"than partially written. Re-embed with the configured model."
                    ),
                    details={"row_index": index, "row_dim": actual, "stored_dim": stored_dim},
                    source="document_repository",
                )

        return None

    @trace_layer("repository")
    async def upsert_chunks(self, rows: list[dict[str, Any]]) -> DocumentResult[None]:
        if not rows:
            return Success(inner_value=None)
        width_error = self._reject_width_mismatch(rows)
        if width_error is not None:
            return Failure(inner_value=width_error)
        try:
            await self.session.execute(build_chunk_upsert_statement(rows))
            return Success(inner_value=None)
        except IntegrityError as exc:
            add_database_error_note(exc, table="chunks")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentChunkConflictError(
                    message="Chunk upsert failed due to a constraint violation",
                    details={"error": str(object=exc)},
                    source="document_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chunks")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while upserting chunks",
                    details={"error": str(object=exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def analyze_chunks(self) -> DocumentResult[None]:
        try:
            await self.session.execute(statement=text(text="ANALYZE chunks"))
            return Success(None)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chunks")
            await self.session.rollback()
            return Failure(
                DocumentDatabaseError(
                    message="Database error while analyzing chunks",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def fetch_status(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> DocumentResult[dict[str, Any] | None]:
        try:
            statement: TextClause = text(
                text="""
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
                WHERE d.user_id = :user_id AND d.id = CAST(:document_id AS uuid)
                GROUP BY d.id, d.status, d.object_uri, d.title, d.document_kind
                """
            )
            result = await self.session.execute(
                statement, params={"user_id": user_id, "document_id": document_id}
            )
            row: RowMapping | None = result.mappings().one_or_none()
            if row is None:
                return Failure(
                    inner_value=DocumentStatusNotFoundError(
                        message="Status not found for the given document",
                        details={"user_id": user_id, "document_id": document_id},
                        source="document_repository",
                    )
                )
            return Success(inner_value=dict(row))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents, chunks")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while fetching document status",
                    details={
                        "user_id": user_id,
                        "document_id": document_id,
                        "error": str(object=exc),
                    },
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def bm25_search(
        self,
        *,
        user_id: str,
        query: str,
        candidate_limit: int,
        filter_params: dict[str, Any],
        bm25_threshold: float | None = None,
        exact_phrase: str | None = None,
    ) -> DocumentResult[list[dict[str, Any]]]:
        try:
            phrase_pattern = _phrase_like_pattern(exact_phrase) if exact_phrase else None
            fetch_limit = (
                candidate_limit * PHRASE_OVERFETCH_MULTIPLE
                if phrase_pattern is not None
                else candidate_limit
            )
            statement: TextClause = text(
                text="""
                SELECT
                    c.id::text AS chunk_id,
                    (-1 * (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx'))) AS score
                FROM chunks AS c
                WHERE c.user_id = :user_id
                  AND (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')) < 0
                  AND (
                    CAST(:bm25_threshold AS double precision) IS NULL
                    OR (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx'))
                        < CAST(:bm25_threshold AS double precision)
                  )
                  AND (
                    CAST(:phrase_pattern AS text) IS NULL
                    OR c.search_text LIKE CAST(:phrase_pattern AS text) ESCAPE '\\'
                  )
                """
                + _FILTER_SQL
                + """
                ORDER BY (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx')) ASC,
                    c.id ASC
                LIMIT :fetch_limit
                """
            )
            result = await self.session.execute(
                statement,
                {
                    "user_id": user_id,
                    "query": query,
                    "candidate_limit": candidate_limit,
                    "bm25_threshold": bm25_threshold,
                    "phrase_pattern": phrase_pattern,
                    "fetch_limit": fetch_limit,
                    **filter_params,
                },
            )
            rows = [dict(row) for row in result.mappings().all()]
            return Success(inner_value=rows[:candidate_limit])
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chunks")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while performing BM25 search",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def vector_search(
        self,
        *,
        user_id: str,
        embedding: list[float],
        candidate_limit: int,
        filter_params: dict[str, Any],
    ) -> DocumentResult[list[dict[str, Any]]]:
        try:
            statement: TextClause = text(
                text="""
                SELECT
                    c.id::text AS chunk_id,
                    (1 - (c.embedding <=> CAST(:embedding AS vector))) AS score
                FROM chunks AS c
                WHERE c.user_id = :user_id
                  AND c.embedding IS NOT NULL
                """
                + _FILTER_SQL
                + """
                ORDER BY c.embedding <=> CAST(:embedding AS vector),
                    c.id ASC
                LIMIT :candidate_limit
                """
            )
            await self.session.execute(
                statement=text(
                    text=f"SET LOCAL diskann.query_search_list_size = {DISKANN_QUERY_SEARCH_LIST_SIZE}"
                )
            )
            await self.session.execute(
                statement=text(text=f"SET LOCAL diskann.query_rescore = {DISKANN_QUERY_RESCORE}")
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
            return Success(inner_value=[dict(row) for row in result.mappings().all()])
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chunks")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while performing vector search",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def trigram_search(
        self,
        *,
        user_id: str,
        query: str,
        candidate_limit: int,
        filter_params: dict[str, Any],
    ) -> DocumentResult[list[dict[str, Any]]]:
        try:
            statement: TextClause = text(
                text="""
                SELECT
                    c.id::text AS chunk_id,
                    similarity(c.search_text, :query) AS score
                FROM chunks AS c
                WHERE c.user_id = :user_id
                  AND c.search_text % :query
                  AND similarity(c.search_text, :query) >= :similarity_threshold
                """
                + _FILTER_SQL
                + """
                ORDER BY similarity(c.search_text, :query) DESC,
                    c.id ASC
                LIMIT :candidate_limit
                """
            )
            result = await self.session.execute(
                statement,
                params={
                    "user_id": user_id,
                    "query": query,
                    "candidate_limit": candidate_limit,
                    "similarity_threshold": TRIGRAM_SIMILARITY_THRESHOLD,
                    **filter_params,
                },
            )
            return Success(inner_value=[dict(row) for row in result.mappings().all()])
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chunks")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while performing trigram search",
                    details={"error": str(object=exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def find_document_ids_by_metadata(
        self,
        *,
        user_id: str,
        jurisdictions: Sequence[str],
        document_kinds: Sequence[str],
        matters: Sequence[str],
        limit: int,
    ) -> DocumentResult[list[str]]:
        """Resolve a document allowlist from cheap metadata, without retrieval.

        A btree/metadata lookup over the documents relation: no vectors, no
        ranking, no index statistics. An empty criteria set matches nothing
        here — the caller treats that as "cannot narrow" and proceeds
        unconstrained.
        """
        if not (jurisdictions or document_kinds or matters):
            return Success(inner_value=[])
        try:
            statement: TextClause = text(
                text="""
                SELECT d.id::text AS document_id
                FROM documents AS d
                WHERE d.user_id = :user_id
                  AND (
                    (
                      CAST(:jurisdictions AS text[]) IS NOT NULL
                      AND d.jurisdiction = ANY(CAST(:jurisdictions AS text[]))
                    )
                    OR (
                      CAST(:document_kinds AS text[]) IS NOT NULL
                      AND d.document_kind = ANY(CAST(:document_kinds AS text[]))
                    )
                    OR (
                      CAST(:matters AS text[]) IS NOT NULL
                      AND d.metadata_->>'matter' = ANY(CAST(:matters AS text[]))
                    )
                  )
                ORDER BY d.id::text ASC
                LIMIT :limit
                """
            )
            result = await self.session.execute(
                statement,
                {
                    "user_id": user_id,
                    "jurisdictions": list(jurisdictions) or None,
                    "document_kinds": list(document_kinds) or None,
                    "matters": list(matters) or None,
                    "limit": limit,
                },
            )
            return Success(inner_value=[str(row[0]) for row in result.all()])
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while resolving source allowlist",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def fetch_chunks_by_ids(
        self, chunk_ids: Sequence[str]
    ) -> DocumentResult[dict[str, dict[str, Any]]]:
        if not chunk_ids:
            return Success({})
        try:
            statement: TextClause = text(
                text="""
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                d.title AS title,
                c.content AS content,
                c.preamble AS preamble,
                c.search_text AS search_text,
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
            result = await self.session.execute(statement, params={"chunk_ids": list(chunk_ids)})
            return Success(
                {str(object=row["chunk_id"]): dict(row) for row in result.mappings().all()}
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chunks, documents")
            await self.session.rollback()
            return Failure(
                DocumentDatabaseError(
                    message="Database error while fetching chunks",
                    details={"error": str(exc)},
                    source="document_repository",
                )
            )


def build_chunk_upsert_statement(rows: list[dict[str, Any]]) -> Insert:
    """The chunk upsert: bulk insert with conflict-resolved refresh.

    Extracted from ``upsert_chunks`` so tests can compile the statement against
    the PostgreSQL dialect without a session. The conflict set must name every
    column a re-ingest may refresh — SQLAlchemy does not merge ``onupdate``
    defaults into an explicit ``DO UPDATE SET``, so anything absent from ``set_``
    keeps its first-written value forever even though the ORM hook exists.
    """
    statement: Insert = insert(table=UnifiedChunk).values(rows)
    return statement.on_conflict_do_update(
        constraint="uq_chunks_document_version_chunk_index",
        set_={
            "chunk_kind": statement.excluded.chunk_kind,
            "content": statement.excluded.content,
            "preamble": statement.excluded.preamble,
            "locus": statement.excluded.locus,
            "clause_type": statement.excluded.clause_type,
            "page_no": statement.excluded.page_no,
            "embedding": statement.excluded.embedding,
            "metadata_": statement.excluded.metadata_,
            "custom_metadata": statement.excluded.custom_metadata,
            "quality_warnings": statement.excluded.quality_warnings,
            "graphiti_episode_id": statement.excluded.graphiti_episode_id,
            "graphiti_verified": statement.excluded.graphiti_verified,
            "updated_at": statement.excluded.updated_at,
            "instrument_name": statement.excluded.instrument_name,
            "section_ref": statement.excluded.section_ref,
            "instrument_year": statement.excluded.instrument_year,
        },
    )


def build_chunk_rows(
    *, document_id: str, user_id: str, chunks: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    now = datetime.now(tz=UTC)
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        locus = chunk.get("locus")
        # Unknown structural position is absent, never "". An empty string would
        # look populated to consumers and collide with "no locus" conventions.
        if not locus:
            locus = None
        rows.append(
            {
                **chunk,
                "document_id": document_id,
                "user_id": user_id,
                "document_version": int(chunk.get("document_version", 1)),
                "locus": locus,
                # The upsert's DO UPDATE SET refreshes updated_at from the row, so
                # every write path carries it; callers may pin their own value.
                "updated_at": chunk.get("updated_at", now),
            }
        )
    return rows


def build_search_filter_params(*, metadata_filter: dict[str, Any]) -> dict[str, Any]:
    document_ids = metadata_filter.get("document_ids") or []
    parties = metadata_filter.get("parties") or []
    chunk_ids = metadata_filter.get("chunk_ids") or []
    return {
        "document_ids": document_ids,
        "chunk_ids": chunk_ids,
        "document_kind": metadata_filter.get("document_kind"),
        "jurisdiction": metadata_filter.get("jurisdiction"),
        "contract_type": metadata_filter.get("contract_type"),
        "clause_type": metadata_filter.get("clause_type"),
        "require_graphiti_verified": bool(metadata_filter.get("require_graphiti_verified")),
        "metadata_filter": json.dumps(metadata_filter.get("metadata_", {})),
        "parties_filter": json.dumps(parties),
    }


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.12f}" for value in embedding) + "]"
