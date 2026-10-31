"""Persistence and retrieval operations for site-search documents/chunks.

Modernized port of the legacy ``logs/search/repository.py``:

- Every method returns ``Result[T, SearchError]``; expected failures are
  ``Failure``s, never raises (EXCEPTION-RULES).
- Every query is tenant-scoped; there is no unscoped read.
- uuid binds cross the asyncpg driver as text plus an explicit CAST via
  ``app.utils.sql_binds`` (``uuid_param`` / ``uuid_any``).
- ``text()`` is used only where the builder cannot go: the pg_textsearch
  ``<@>`` operator with a literal index name, the DiskANN ``<=>`` operator and
  its ``SET LOCAL diskann.*`` session GUCs, pg_trgm's ``%`` operator and
  ``similarity()`` ordering, and the ``ANALYZE`` maintenance command — each
  with a documented comment, mirroring
  ``features/documents/adapters/persistence/postgres/document_repository.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from returns.result import Failure, Success
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.config import get_settings
from app.features.search.adapters.persistence.models import SearchChunk, SearchDocument
from app.features.search.domain.errors import (
    SearchConflictError,
    SearchInfrastructureError,
)
from app.kernel.result import add_database_error_note
from app.retrieval.fusion import RankedResultRow
from app.utils.sql_binds import uuid_any, uuid_param

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import TextClause

    from app.features.search.domain.errors import SearchResult

# Shared tenant + metadata-containment predicate appended to every retrieval
# branch. The containment half is the documented text() JSONB form: the
# `{}`-empty guard lets one statement serve both filtered and unfiltered calls,
# exactly like the documents repository's `_FILTER_SQL`.
_FILTER_SQL = """
  AND c.tenant_id = :tenant_id
  AND (
    CAST(:metadata_filter AS text) = '{}'
    OR c.chunk_metadata @> CAST(:metadata_filter AS jsonb)
  )
"""


class SearchRepository:
    """Database operations for the site-search feature."""

    def __init__(self, session: AsyncSession):
        self.session: AsyncSession = session

    async def get_document_by_content_hash(
        self, *, tenant_id: str, content_hash: str
    ) -> SearchResult[SearchDocument | None]:
        """Tenant-scoped dedupe lookup; ``None`` is the normal first-ingest path."""
        try:
            statement = select(SearchDocument).where(
                SearchDocument.tenant_id == tenant_id,
                SearchDocument.content_hash == content_hash,
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_documents")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while fetching search document by content hash",
                    details={"content_hash": content_hash},
                    source="search_repository",
                )
            )

    async def get_document_by_id(
        self, *, tenant_id: str, document_id: str
    ) -> SearchResult[SearchDocument | None]:
        """Tenant-scoped document lookup by uuid string."""
        try:
            statement = select(SearchDocument).where(
                SearchDocument.tenant_id == tenant_id,
                SearchDocument.id == uuid_param("document_id", document_id),
            )
            result = await self.session.execute(statement)
            return Success(result.scalar_one_or_none())
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_documents")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while fetching search document by ID",
                    details={"document_id": document_id},
                    source="search_repository",
                )
            )

    async def create_document(
        self,
        *,
        tenant_id: str,
        title: str,
        source_uri: str | None,
        content_hash: str,
        doc_metadata: dict[str, object],
    ) -> SearchResult[SearchDocument]:
        try:
            document = SearchDocument(
                tenant_id=tenant_id,
                title=title,
                source_uri=source_uri,
                content_hash=content_hash,
                doc_metadata=doc_metadata,
            )
            self.session.add(document)
            await self.session.flush()
            return Success(document)
        except IntegrityError as exc:
            add_database_error_note(exc, table="search_documents")
            await self.session.rollback()
            return Failure(
                SearchConflictError(
                    message="Search document creation failed due to a constraint violation",
                    details={"content_hash": content_hash},
                    source="search_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_documents")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while creating search document",
                    details={"content_hash": content_hash},
                    source="search_repository",
                )
            )

    async def set_document_status(self, *, document_id: str, status: str) -> SearchResult[None]:
        """Worker-side lifecycle update; the worker owns the tenant it acts for."""
        try:
            document = await self.session.get(SearchDocument, UUID(document_id))
            if document is not None:
                document.status = status
                await self.session.flush()
            return Success(None)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_documents")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while updating search document status",
                    details={"document_id": document_id, "status": status},
                    source="search_repository",
                )
            )

    async def rollback(self) -> None:
        """Discard the caller-visible pending writes on this session.

        Ingest stages the document row and the outbox event on the request
        session, which the ``get_postgres_db`` dependency commits after the
        handler returns. When the service returns a ``Failure`` after either
        was flushed, it must unwind them here first — a returned ``Failure``
        does not roll a request-scoped session back, and committing a
        document without its job strands it as a permanent pending orphan.
        """
        await self.session.rollback()

    async def upsert_chunks(self, rows: list[dict[str, Any]]) -> SearchResult[None]:
        """Bulk upsert chunk rows on the (document_id, chunk_index) key."""
        if not rows:
            return Success(None)
        try:
            await self.session.execute(build_chunk_upsert_statement(rows))
            return Success(None)
        except IntegrityError as exc:
            add_database_error_note(exc, table="search_chunks")
            await self.session.rollback()
            return Failure(
                SearchConflictError(
                    message="Chunk upsert failed due to a constraint violation",
                    details={"row_count": len(rows)},
                    source="search_repository",
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_chunks")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while upserting search chunks",
                    details={"row_count": len(rows)},
                    source="search_repository",
                )
            )

    async def analyze_chunks(self) -> SearchResult[None]:
        try:
            # text() is the only representation of a maintenance command
            # (ANALYZE); it is not a query and gains nothing from expressions.
            await self.session.execute(statement=text(text="ANALYZE search_chunks"))
            return Success(None)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_chunks")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while analyzing search_chunks",
                    source="search_repository",
                )
            )

    async def bm25_search(
        self,
        *,
        tenant_id: str,
        query: str,
        candidate_limit: int,
        metadata_filter: dict[str, object],
    ) -> SearchResult[list[RankedResultRow]]:
        try:
            # Documented text() fallback (see app/utils/sql_binds.py): this query
            # ranks through pg_textsearch's `<@>` operator and a literal index
            # name, both pinned by source gates (bm25 sign convention, index
            # identity). The index name is deployment configuration
            # (SEARCH_CHUNKS_BM25_INDEX_NAME), never user input, and
            # to_bm25query's second argument is a literal, not a bindable
            # parameter. Expression-fying it would trade a reviewed literal
            # contract for op() soup with zero safety gain.
            statement: TextClause = text(
                text=f"""
                SELECT
                    c.id::text AS chunk_id,
                    (-1 * (c.content <@> to_bm25query(:query, '{_bm25_index_name()}'))) AS score
                FROM search_chunks AS c
                WHERE (c.content <@> to_bm25query(:query, '{_bm25_index_name()}')) < 0
                """  # noqa: S608 — the only interpolation is the settings-pinned index name, validated to a plain identifier by _bm25_index_name
                + _FILTER_SQL
                + f"""
                ORDER BY (c.content <@> to_bm25query(:query, '{_bm25_index_name()}')) ASC,
                    c.id ASC
                LIMIT :candidate_limit
                """
            )
            result = await self.session.execute(
                statement,
                {
                    "query": query,
                    "candidate_limit": candidate_limit,
                    "metadata_filter": json.dumps(metadata_filter),
                    "tenant_id": tenant_id,
                },
            )
            return Success(_rank_rows(result.mappings().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_chunks")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while performing BM25 search",
                    source="search_repository",
                )
            )

    async def vector_search(
        self,
        *,
        tenant_id: str,
        embedding: list[float],
        candidate_limit: int,
        metadata_filter: dict[str, object],
    ) -> SearchResult[list[RankedResultRow]]:
        settings = get_settings()
        search_list_size = settings.SEARCH_DISKANN_QUERY_SEARCH_LIST_SIZE
        rescore = settings.SEARCH_DISKANN_QUERY_RESCORE
        try:
            # Documented text() fallback: the DiskANN cosine operator `<=>` and
            # the `SET LOCAL diskann.*` session GUCs have no expression form.
            # The GUC values are integer settings fields, never user input.
            statement: TextClause = text(
                text="""
                SELECT
                    c.id::text AS chunk_id,
                    (1 - (c.embedding <=> CAST(:embedding AS vector))) AS score
                FROM search_chunks AS c
                WHERE c.embedding IS NOT NULL
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
                    text=f"SET LOCAL diskann.query_search_list_size = {search_list_size}"
                )
            )
            await self.session.execute(
                statement=text(text=f"SET LOCAL diskann.query_rescore = {rescore}")
            )
            result = await self.session.execute(
                statement,
                {
                    "embedding": _vector_literal(embedding),
                    "candidate_limit": candidate_limit,
                    "metadata_filter": json.dumps(metadata_filter),
                    "tenant_id": tenant_id,
                },
            )
            return Success(_rank_rows(result.mappings().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_chunks")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while performing vector search",
                    source="search_repository",
                )
            )

    async def trigram_search(
        self,
        *,
        tenant_id: str,
        query: str,
        candidate_limit: int,
        metadata_filter: dict[str, object],
    ) -> SearchResult[list[RankedResultRow]]:
        settings = get_settings()
        similarity_threshold = settings.SEARCH_TRIGRAM_SIMILARITY_THRESHOLD
        try:
            # Documented text() fallback: pg_trgm's `%` match operator and
            # `similarity()` ordering have no readable expression form.
            statement: TextClause = text(
                text="""
                SELECT
                    c.id::text AS chunk_id,
                    similarity(c.content, :query) AS score
                FROM search_chunks AS c
                WHERE c.content % :query
                  AND similarity(c.content, :query) >= :similarity_threshold
                """
                + _FILTER_SQL
                + """
                ORDER BY similarity(c.content, :query) DESC,
                    c.id ASC
                LIMIT :candidate_limit
                """
            )
            # The `%` operator filters against the *session* similarity
            # threshold (PostgreSQL default 0.3), not this query's bind — so a
            # configured 0.1 would silently lose the 0.1-0.3 band before the
            # explicit similarity() check ever ran. `set_config(..., is_local
            # => true)` is the parameterizable equivalent of `SET LOCAL` (a
            # utility statement accepts no bind parameters), transaction-
            # scoped exactly like the DiskANN GUCs above; the value is a
            # settings float, never user input.
            await self.session.execute(
                statement=text(
                    text="SELECT set_config('pg_trgm.similarity_threshold', :threshold, true)"
                ),
                params={"threshold": str(similarity_threshold)},
            )
            result = await self.session.execute(
                statement,
                params={
                    "query": query,
                    "candidate_limit": candidate_limit,
                    "similarity_threshold": similarity_threshold,
                    "metadata_filter": json.dumps(metadata_filter),
                    "tenant_id": tenant_id,
                },
            )
            return Success(_rank_rows(result.mappings().all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_chunks")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while performing trigram search",
                    source="search_repository",
                )
            )

    async def fetch_chunks_by_ids(
        self, *, tenant_id: str, chunk_ids: Sequence[str]
    ) -> SearchResult[dict[str, dict[str, Any]]]:
        """Hydrate fused chunk ids, tenant-scoped; missing ids are simply absent."""
        if not chunk_ids:
            return Success({})
        try:
            # Expression form of the keyed chunk fetch: search_chunks JOIN
            # search_documents with a uuid[] ANY filter and a tenant pin.
            statement = (
                select(
                    SearchChunk.id.label("chunk_id"),
                    SearchChunk.document_id.label("document_id"),
                    SearchChunk.chunk_index,
                    SearchChunk.content,
                    SearchChunk.chunk_metadata,
                    SearchDocument.title,
                )
                .join_from(
                    SearchChunk,
                    SearchDocument,
                    SearchDocument.id == SearchChunk.document_id,
                )
                .where(
                    SearchChunk.tenant_id == tenant_id,
                    SearchChunk.id == uuid_any("chunk_ids", list(chunk_ids)),
                )
            )
            result = await self.session.execute(statement)
            return Success(
                {
                    str(row["chunk_id"]): {
                        "chunk_id": str(row["chunk_id"]),
                        "document_id": str(row["document_id"]),
                        "chunk_index": int(row["chunk_index"]),
                        "content": str(row["content"]),
                        "chunk_metadata": dict(row["chunk_metadata"] or {}),
                        "title": str(row["title"]),
                    }
                    for row in result.mappings().all()
                }
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="search_chunks, search_documents")
            await self.session.rollback()
            return Failure(
                SearchInfrastructureError(
                    message="Database error while fetching search chunks",
                    source="search_repository",
                )
            )


def _bm25_index_name() -> str:
    """The configured bm25 index name, validated to a safe identifier shape.

    It is interpolated into ``to_bm25query``'s literal (the operator takes an
    index name, not a parameter). The guard keeps a mistyped settings value a
    startup/compile-time failure instead of an injection surface: only
    alphanumerics and underscores are allowed through.
    """
    name = get_settings().SEARCH_CHUNKS_BM25_INDEX_NAME
    if not name or not name.replace("_", "").isalnum():
        msg = f"SEARCH_CHUNKS_BM25_INDEX_NAME must be a plain identifier, got {name!r}"
        raise ValueError(msg)
    return name


def _rank_rows(rows: Sequence[Any]) -> list[RankedResultRow]:
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
    return "[" + ",".join(f"{value:.12f}" for value in embedding) + "]"


def build_chunk_upsert_statement(rows: list[dict[str, Any]]) -> Any:
    """The search chunk upsert, extracted for statement-compilation tests.

    SQLAlchemy does not merge ``onupdate`` defaults into an explicit
    ``DO UPDATE SET``, so every re-ingest refresh column is named in ``set_``.
    """
    statement = insert(SearchChunk).values(rows)
    return statement.on_conflict_do_update(
        constraint="uq_search_chunks_document_chunk_index",
        set_={
            "content": statement.excluded.content,
            "embedding": statement.excluded.embedding,
            "chunk_metadata": statement.excluded.chunk_metadata,
            "updated_at": statement.excluded.updated_at,
        },
    )


def build_chunk_rows(
    *,
    document_id: str,
    tenant_id: str,
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create row payloads for repository upsert calls."""
    now = datetime.now(tz=UTC)
    return [
        {
            "id": UUID(str(chunk["id"])),
            "document_id": UUID(document_id),
            "tenant_id": tenant_id,
            "chunk_index": int(chunk["chunk_index"]),
            "content": str(chunk["content"]),
            "embedding": chunk["embedding"],
            "chunk_metadata": dict(chunk["chunk_metadata"]),
            "created_at": now,
            "updated_at": now,
        }
        for chunk in chunks
    ]
