"""Persistence and retrieval operations for unified documents/chunks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from returns.result import Failure, Success
from sqlalchemy import (
    Boolean,
    Integer,
    String,
    and_,
    bindparam,
    case,
    cast,
    func,
    literal_column,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.features.auth.domain.tenancy import personal_tenant_id
from app.features.documents.domain.chunk import CHUNK_EMBEDDING_DIM, UnifiedChunk
from app.features.documents.domain.constants import (
    CANONICAL_PIPELINE_VERSION,
    DISKANN_QUERY_RESCORE,
    DISKANN_QUERY_SEARCH_LIST_SIZE,
    PHRASE_OVERFETCH_MULTIPLE,
    TRIGRAM_SIMILARITY_THRESHOLD,
)
from app.features.documents.domain.document import UnifiedDocument
from app.features.documents.domain.errors import (
    DocumentChunkConflictError,
    DocumentConflictError,
    DocumentDatabaseError,
    DocumentEmbeddingWidthError,
    DocumentNotFoundError,
    DocumentStatusNotFoundError,
)
from app.kernel.result import add_database_error_note
from app.observability.logging import trace_layer
from app.utils.embedding import stored_width_mismatch, width_mismatch_detail
from app.utils.sql_binds import uuid_any, uuid_param

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from sqlalchemy.dialects.postgresql.dml import Insert
    from sqlalchemy.engine.result import Result
    from sqlalchemy.engine.row import RowMapping
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import ColumnElement, TextClause
    from sqlalchemy.sql.selectable import Select

    from app.features.documents.domain.errors import DocumentResult

_FILTER_SQL = """
  AND c.tenant_id = :tenant_id
  AND (CAST(:scope_bound AS boolean) IS FALSE OR c.document_id = ANY(CAST(:document_ids AS uuid[])))
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
    CAST(:contract_index AS integer) IS NULL
    OR c.contract_index = CAST(:contract_index AS integer)
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


def _resolve_tenant_id(tenant_id: str | None, user_id: str) -> str:
    """Resolve the migration-only one-user tenant compatibility boundary.

    Current HTTP/worker call paths pass an authoritative tenant.  Keeping the
    fallback in this one named helper preserves old repository test doubles and
    migration jobs without hiding an implicit tenant choice in every query.
    """
    return tenant_id if tenant_id is not None else personal_tenant_id(user_id)


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
    async def resolve_scope_versions(
        self,
        *,
        tenant_id: str,
        document_ids: Sequence[str],
    ) -> DocumentResult[tuple[tuple[str, int, str], ...]]:
        """Resolve the corpus identity used by response caches.

        Cache lookup must happen after this query: re-ingesting a document can
        leave the same document id in the allowlist while replacing its chunks.
        The version tuple therefore acts as the content-addressed read fence.
        Missing documents are retained as ``(id, 0, pipeline)`` only by callers
        that choose to represent them; this repository returns rows that exist.
        """
        if not document_ids:
            return Success(())
        try:
            # Expression form of: SELECT c.document_id::text,
            # COALESCE(MAX(c.document_version), 1)::int,
            # COALESCE(MAX(c.pipeline_version), :pipeline_version) FROM chunks
            # WHERE tenant_id AND document_id = ANY(uuid[]) GROUP BY/ORDER BY document_id.
            statement = (
                select(
                    cast(UnifiedChunk.document_id, String).label("document_id"),
                    cast(
                        func.coalesce(func.max(UnifiedChunk.document_version), 1),
                        Integer,
                    ).label("document_version"),
                    func.coalesce(
                        func.max(UnifiedChunk.pipeline_version), CANONICAL_PIPELINE_VERSION
                    ).label("pipeline_version"),
                )
                .where(
                    UnifiedChunk.tenant_id == tenant_id,
                    UnifiedChunk.document_id == uuid_any("document_ids", document_ids),
                )
                .group_by(UnifiedChunk.document_id)
                .order_by(UnifiedChunk.document_id)
            )

            result = await self.session.execute(statement)
            return Success(
                tuple(
                    (
                        str(row["document_id"]),
                        int(row["document_version"]),
                        str(row["pipeline_version"]),
                    )
                    for row in result.mappings().all()
                )
            )
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chunks")
            await self.session.rollback()
            return Failure(
                DocumentDatabaseError(
                    message="Database error while resolving document versions",
                    details={"tenant_id": tenant_id, "document_ids": list(document_ids)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def resolve_scope_contracts(
        self,
        *,
        tenant_id: str,
        document_ids: Sequence[str],
    ) -> DocumentResult[tuple[tuple[str, int, tuple[int | None, ...], bool], ...]]:
        """Resolve contract partitions for each authorized document version.

        ``NULL`` is returned as a legacy marker only when every chunk in that
        version has a null contract index.  Mixed modern/null data is treated
        as modern and the null partition is excluded from Graphiti reads.
        """
        if not document_ids:
            return Success(())
        try:
            # Expression form of the chunk-scope aggregate: GROUP BY
            # (document_id, document_version) with array_agg(DISTINCT
            # contract_index) and bool_and(contract_index IS NULL), tenant- and
            # allowlist-scoped via `document_id = ANY(CAST(:ids AS uuid[]))`.
            statement = (
                select(
                    cast(UnifiedChunk.document_id, String).label("document_id"),
                    cast(UnifiedChunk.document_version, Integer).label("document_version"),
                    func.array_agg(func.distinct(UnifiedChunk.contract_index)).label(
                        "contract_indexes"
                    ),
                    func.bool_and(UnifiedChunk.contract_index.is_(None)).label("all_legacy_null"),
                )
                .where(
                    UnifiedChunk.tenant_id == tenant_id,
                    UnifiedChunk.document_id == uuid_any("document_ids", document_ids),
                )
                .group_by(UnifiedChunk.document_id, UnifiedChunk.document_version)
                .order_by(UnifiedChunk.document_id, UnifiedChunk.document_version)
            )
            result = await self.session.execute(statement)
            scopes: list[tuple[str, int, tuple[int | None, ...], bool]] = []
            for row in result.mappings().all():
                raw_indexes = row["contract_indexes"] or []
                indexes = tuple(
                    int(index) for index in raw_indexes if index is not None and int(index) >= 0
                )
                legacy_null = bool(row["all_legacy_null"]) and not indexes
                scopes.append(
                    (str(row["document_id"]), int(row["document_version"]), indexes, legacy_null)
                )
            return Success(tuple(scopes))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="chunks")
            await self.session.rollback()
            return Failure(
                DocumentDatabaseError(
                    message="Database error while resolving document contract scopes",
                    details={"tenant_id": tenant_id, "document_ids": list(document_ids)},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def get_document_by_user_hash(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        content_hash: str,
    ) -> DocumentResult[UnifiedDocument | None]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        try:
            statement: Select[tuple[UnifiedDocument]] = select(UnifiedDocument).where(
                UnifiedDocument.tenant_id == tenant_id,
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
        tenant_id: str | None = None,
    ) -> DocumentResult[UnifiedDocument | None]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        try:
            statement: Select[tuple[UnifiedDocument]] = select(UnifiedDocument).where(
                UnifiedDocument.tenant_id == tenant_id,
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
        tenant_id: str | None = None,
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
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        try:
            document = UnifiedDocument(
                user_id=user_id,
                tenant_id=tenant_id,
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
        ingestion_identity: dict[str, object] | None = None,
    ) -> DocumentResult[None]:
        try:
            identity_tenant = (
                str(ingestion_identity.get("tenant_id", ""))
                if ingestion_identity is not None
                else None
            )
            identity_version = (
                int(raw_version)
                if ingestion_identity is not None
                and isinstance(
                    raw_version := ingestion_identity.get("document_version"), (int, str)
                )
                and not isinstance(raw_version, bool)
                else None
            )
            identity_attempt = (
                int(raw_attempt)
                if ingestion_identity is not None
                and isinstance(raw_attempt := ingestion_identity.get("attempt"), (int, str))
                and not isinstance(raw_attempt, bool)
                else None
            )
            identity_task_id = (
                str(ingestion_identity.get("task_id") or "")
                if ingestion_identity is not None
                else None
            )
            updated_at = datetime.now(tz=UTC)
            # Expression form of the guarded status UPDATE. `metadata_json` is a
            # single named bind used twice, mirroring the original's single
            # :metadata_ parameter inside both the CASE guard and the jsonb `||`
            # merge. The identity guard keeps its NULL semantics through
            # bindparam: a NULL :identity_version short-circuits the guard to
            # TRUE exactly like `CAST(:identity_version AS integer) IS NULL`.
            metadata_json = cast(
                bindparam(
                    "metadata_",
                    value=json.dumps(metadata_) if metadata_ is not None else None,
                    type_=String,
                ),
                JSONB,
            )
            identity_version_bind = bindparam(
                "identity_version", value=identity_version, type_=Integer
            )
            statement = (
                update(UnifiedDocument)
                .where(
                    UnifiedDocument.id == uuid_param("document_id", document_id),
                    or_(
                        identity_version_bind.is_(None),
                        and_(
                            UnifiedDocument.metadata_["ingestion_state"]["tenant_id"].astext
                            == bindparam("identity_tenant", value=identity_tenant, type_=String),
                            cast(
                                UnifiedDocument.metadata_["ingestion_state"][
                                    "document_version"
                                ].astext,
                                Integer,
                            )
                            == identity_version_bind,
                            cast(
                                UnifiedDocument.metadata_["ingestion_state"]["attempt"].astext,
                                Integer,
                            )
                            == bindparam("identity_attempt", value=identity_attempt, type_=Integer),
                            UnifiedDocument.metadata_["ingestion_state"]["task_id"].astext
                            == bindparam("identity_task_id", value=identity_task_id, type_=String),
                        ),
                    ),
                )
                .values(
                    status=status,
                    title=func.coalesce(
                        bindparam("title", value=title, type_=String), UnifiedDocument.title
                    ),
                    document_kind=func.coalesce(
                        bindparam("document_kind", value=document_kind, type_=String),
                        UnifiedDocument.document_kind,
                    ),
                    jurisdiction=func.coalesce(
                        bindparam("jurisdiction", value=jurisdiction, type_=String),
                        UnifiedDocument.jurisdiction,
                    ),
                    contract_type=func.coalesce(
                        bindparam("contract_type", value=contract_type, type_=String),
                        UnifiedDocument.contract_type,
                    ),
                    parties=func.coalesce(
                        cast(
                            bindparam(
                                "parties",
                                value=json.dumps(parties) if parties is not None else None,
                                type_=String,
                            ),
                            JSONB,
                        ),
                        UnifiedDocument.parties,
                    ),
                    metadata_=case(
                        (metadata_json.is_(None), UnifiedDocument.metadata_),
                        else_=func.coalesce(
                            UnifiedDocument.metadata_, literal_column("'{}'::jsonb")
                        ).op("||")(metadata_json),
                    ),
                    structural_tree=func.coalesce(
                        cast(
                            bindparam(
                                "structural_tree",
                                value=(
                                    json.dumps(structural_tree)
                                    if structural_tree is not None
                                    else None
                                ),
                                type_=String,
                            ),
                            JSONB,
                        ),
                        UnifiedDocument.structural_tree,
                    ),
                    extraction_incomplete=func.coalesce(
                        bindparam(
                            "extraction_incomplete", value=extraction_incomplete, type_=Boolean
                        ),
                        UnifiedDocument.extraction_incomplete,
                    ),
                    updated_at=updated_at,
                )
                .execution_options(synchronize_session=False)
            )
            await self.session.execute(statement)
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
    async def apply_terminal_ingestion_failure(
        self,
        *,
        document_id: str,
        tenant_id: str,
        document_version: int,
        attempt: int,
        task_id: str,
        metadata_: dict[str, object],
    ) -> DocumentResult[bool]:
        """Apply a dead-letter failure only to the current ingestion identity.

        The row lock makes the read/compare/write one transaction.  This is
        deliberately separate from the general status updater: terminal
        notifications are at-least-once and may arrive after a retry,
        re-ingestion, or successful completion.
        """
        try:
            result = await self.session.execute(
                select(
                    UnifiedDocument.tenant_id,
                    UnifiedDocument.status,
                    UnifiedDocument.metadata_,
                )
                .where(UnifiedDocument.id == uuid_param("document_id", document_id))
                .with_for_update()
            )
            row = result.mappings().one_or_none()
            if row is None or str(row["tenant_id"]) != tenant_id:
                return Success(False)
            current_status = str(row["status"] or "")
            current_metadata = row["metadata_"] if isinstance(row["metadata_"], dict) else {}
            raw_state = current_metadata.get("ingestion_state")
            if current_status in {"completed", "completed_with_warnings"}:
                return Success(False)
            if isinstance(raw_state, dict):
                try:
                    stored_tenant = str(raw_state.get("tenant_id", ""))
                    stored_version = int(raw_state.get("document_version", 0))
                    stored_attempt = int(raw_state.get("attempt", -1))
                    stored_task_id = str(raw_state.get("task_id") or "")
                    identity_matches = (
                        stored_tenant == tenant_id
                        and stored_version == document_version
                        and stored_task_id == task_id
                        and stored_attempt <= attempt
                    )
                except (TypeError, ValueError):
                    identity_matches = False
                    stored_tenant = ""
                    stored_version = 0
                    stored_attempt = -1
                    stored_task_id = ""
                if not identity_matches:
                    return Success(False)
                if current_status == "failed":
                    return Success(False)
            expected_identity = (
                {
                    "tenant_id": stored_tenant,
                    "document_version": stored_version,
                    "attempt": stored_attempt,
                    "task_id": stored_task_id,
                }
                if isinstance(raw_state, dict)
                else None
            )
            update_result = await self.update_document_status(
                document_id=document_id,
                status="failed",
                metadata_=metadata_,
                extraction_incomplete=True,
                ingestion_identity=expected_identity,
            )
            if isinstance(update_result, Failure):
                return Failure(update_result.failure())
            return Success(True)
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                DocumentDatabaseError(
                    message="Database error while applying terminal ingestion failure",
                    details={"document_id": document_id},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def fetch_structural_trees(
        self,
        *,
        user_id: str,
        document_ids: list[str],
        tenant_id: str | None = None,
    ) -> DocumentResult[list[dict[str, Any]]]:
        """Load tenant-scoped persisted trees for structural navigation."""
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        try:
            # Expression form of: SELECT id::text, structural_tree FROM documents
            # WHERE tenant_id AND structural_tree <> '{}'::jsonb AND (empty
            # allowlist OR id = ANY(uuid[])) ORDER BY id. The cardinality() = 0
            # disjunct becomes a Python-side conditional: an empty list adds no
            # id predicate, which is exactly `id = ANY('{}')` → FALSE dropped
            # from a TRUE constant.
            statement = select(
                cast(UnifiedDocument.id, String).label("document_id"),
                UnifiedDocument.structural_tree,
            ).where(
                UnifiedDocument.tenant_id == tenant_id,
                UnifiedDocument.structural_tree != {},
            )
            if document_ids:
                statement = statement.where(
                    UnifiedDocument.id == uuid_any("document_ids", document_ids)
                )
            result = await self.session.execute(statement.order_by(UnifiedDocument.id))
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
            # text() is the only representation of a maintenance command
            # (ANALYZE); it is not a query and gains nothing from expressions.
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
        tenant_id: str | None = None,
    ) -> DocumentResult[dict[str, Any] | None]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        try:
            # Expression form of the status rollup: documents LEFT JOIN chunks
            # with count/count-FILTER/max aggregates and a jsonb_agg FILTER
            # coalesced to '[]'::jsonb, grouped by the document columns.
            statement = (
                select(
                    cast(UnifiedDocument.id, String).label("document_id"),
                    UnifiedDocument.status,
                    UnifiedDocument.object_uri,
                    UnifiedDocument.title,
                    UnifiedDocument.document_kind,
                    UnifiedDocument.metadata_.label("document_metadata"),
                    cast(func.count(UnifiedChunk.id), Integer).label("chunk_count"),
                    cast(
                        func.count().filter(UnifiedChunk.graphiti_verified),
                        Integer,
                    ).label("verified_chunk_count"),
                    cast(func.max(UnifiedChunk.document_version), Integer).label(
                        "document_version"
                    ),
                    cast(func.max(UnifiedChunk.contract_index), Integer).label("contract_index"),
                    func.max(UnifiedChunk.pipeline_version).label("pipeline_version"),
                    func.max(UnifiedChunk.graph_indexed_at).label("graph_indexed_at"),
                    UnifiedDocument.extraction_incomplete,
                    func.coalesce(
                        func.jsonb_agg(UnifiedChunk.quality_warnings).filter(
                            UnifiedChunk.id.is_not(None)
                        ),
                        literal_column("'[]'::jsonb"),
                    ).label("warnings"),
                )
                .join_from(
                    UnifiedDocument,
                    UnifiedChunk,
                    UnifiedChunk.document_id == UnifiedDocument.id,
                    isouter=True,
                )
                .where(
                    UnifiedDocument.tenant_id == tenant_id,
                    UnifiedDocument.id == uuid_param("document_id", document_id),
                )
                .group_by(
                    UnifiedDocument.id,
                    UnifiedDocument.status,
                    UnifiedDocument.object_uri,
                    UnifiedDocument.title,
                    UnifiedDocument.document_kind,
                    UnifiedDocument.metadata_,
                    UnifiedDocument.extraction_incomplete,
                )
            )
            result = await self.session.execute(statement)
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
        tenant_id: str | None = None,
        query: str,
        candidate_limit: int,
        filter_params: dict[str, Any],
        bm25_threshold: float | None = None,
        exact_phrase: str | None = None,
    ) -> DocumentResult[list[dict[str, Any]]]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        try:
            phrase_pattern = _phrase_like_pattern(exact_phrase) if exact_phrase else None
            fetch_limit = (
                candidate_limit * PHRASE_OVERFETCH_MULTIPLE
                if phrase_pattern is not None
                else candidate_limit
            )
            # Documented text() fallback (see app/utils/sql_binds.py): this query
            # ranks through pg_textsearch's `<@>` operator and a literal index
            # name, both pinned by source gates (bm25 sign convention, index
            # identity). Expression-fying it would trade a reviewed literal
            # contract for op() soup with zero safety gain.
            statement: TextClause = text(
                text="""
                SELECT
                    c.id::text AS chunk_id,
                    (-1 * (c.search_text <@> to_bm25query(:query, 'chunks_bm25_idx'))) AS score
                FROM chunks AS c
                WHERE c.tenant_id = :tenant_id
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
                    "query": query,
                    "candidate_limit": candidate_limit,
                    "bm25_threshold": bm25_threshold,
                    "phrase_pattern": phrase_pattern,
                    "fetch_limit": fetch_limit,
                    **filter_params,
                    "tenant_id": tenant_id,
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
        tenant_id: str | None = None,
        embedding: list[float],
        candidate_limit: int,
        filter_params: dict[str, Any],
    ) -> DocumentResult[list[dict[str, Any]]]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        try:
            # Documented text() fallback: the DiskANN cosine operator `<=>` and
            # the `SET LOCAL diskann.*` session GUCs have no expression form.
            statement: TextClause = text(
                text="""
                SELECT
                    c.id::text AS chunk_id,
                    (1 - (c.embedding <=> CAST(:embedding AS vector))) AS score
                FROM chunks AS c
                WHERE c.tenant_id = :tenant_id
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
                    "embedding": _vector_literal(embedding),
                    "candidate_limit": candidate_limit,
                    **filter_params,
                    "tenant_id": tenant_id,
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
        tenant_id: str | None = None,
        query: str,
        candidate_limit: int,
        filter_params: dict[str, Any],
    ) -> DocumentResult[list[dict[str, Any]]]:
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        try:
            # Documented text() fallback: pg_trgm's `%` match operator and
            # `similarity()` ordering have no readable expression form.
            statement: TextClause = text(
                text="""
                SELECT
                    c.id::text AS chunk_id,
                    similarity(c.search_text, :query) AS score
                FROM chunks AS c
                WHERE c.tenant_id = :tenant_id
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
                    "query": query,
                    "candidate_limit": candidate_limit,
                    "similarity_threshold": TRIGRAM_SIMILARITY_THRESHOLD,
                    **filter_params,
                    "tenant_id": tenant_id,
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
        tenant_id: str | None = None,
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
        tenant_id = _resolve_tenant_id(tenant_id, user_id)
        if not (jurisdictions or document_kinds or matters):
            return Success(inner_value=[])
        try:
            # Expression form of: SELECT d.id::text FROM documents WHERE
            # tenant_id AND (jurisdiction = ANY(text[]) OR document_kind =
            # ANY(text[]) OR metadata_->>'matter' = ANY(text[])) ORDER BY
            # d.id::text LIMIT :limit. Each array predicate was guarded by
            # `CAST(:x AS text[]) IS NOT NULL`; the guard becomes a
            # Python-side conditional on the (already known) non-None list.
            scope_conditions: list[ColumnElement[bool]] = []
            if jurisdictions:
                scope_conditions.append(UnifiedDocument.jurisdiction.in_(jurisdictions))
            if document_kinds:
                scope_conditions.append(UnifiedDocument.document_kind.in_(document_kinds))
            if matters:
                scope_conditions.append(UnifiedDocument.metadata_["matter"].astext.in_(matters))
            statement = (
                select(cast(UnifiedDocument.id, String).label("document_id"))
                .where(
                    UnifiedDocument.tenant_id == tenant_id,
                    or_(*scope_conditions),
                )
                .order_by(cast(UnifiedDocument.id, String).asc())
                .limit(limit)
            )
            result = await self.session.execute(statement)
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
    async def list_document_ids_for_tenant(
        self, *, tenant_id: str
    ) -> DocumentResult[tuple[str, ...]]:
        """Return the server-owned document capability for one tenant.

        This query is deliberately separate from retrieval.  It establishes
        the allowlist used to intersect client filters before any chunk read;
        tenant membership remains the authority and no user-supplied IDs are
        ever used to populate it.  The result is intentionally unbounded: a
        truncated allowlist would reject otherwise valid tenant documents.
        """
        try:
            statement = (
                select(cast(UnifiedDocument.id, String).label("document_id"))
                .where(UnifiedDocument.tenant_id == tenant_id)
                .order_by(cast(UnifiedDocument.id, String).asc())
            )
            result = await self.session.execute(statement)
            return Success(tuple(str(row[0]) for row in result.all()))
        except SQLAlchemyError as exc:
            add_database_error_note(exc, table="documents")
            await self.session.rollback()
            return Failure(
                inner_value=DocumentDatabaseError(
                    message="Database error while resolving tenant document capability",
                    details={"tenant_id": tenant_id},
                    source="document_repository",
                )
            )

    @trace_layer("repository")
    async def fetch_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        exact_phrase: str | None = None,
        tenant_id: str | None = None,
        allowed_doc_ids: Sequence[str] | None = None,
    ) -> DocumentResult[dict[str, dict[str, Any]]]:
        if not chunk_ids:
            return Success({})
        try:
            # Expression form of the keyed chunk fetch: chunks JOIN documents
            # with an allowlist-scoped uuid[] ANY filter, optional tenant pin,
            # and an optional escaped-ILIKE phrase post-filter. The three
            # `CAST(:x AS text) IS NULL OR ...` guards become Python-side
            # conditionals on values the caller already resolved.
            statement = (
                select(
                    cast(UnifiedChunk.id, String).label("chunk_id"),
                    cast(UnifiedChunk.document_id, String).label("document_id"),
                    UnifiedDocument.title,
                    UnifiedChunk.content,
                    UnifiedChunk.preamble,
                    UnifiedChunk.search_text,
                    UnifiedChunk.document_version,
                    UnifiedChunk.contract_index,
                    UnifiedChunk.pipeline_version,
                    UnifiedChunk.chunk_index,
                    UnifiedChunk.chunk_kind,
                    UnifiedChunk.clause_type,
                    UnifiedChunk.page_no,
                    UnifiedChunk.metadata_.label("chunk_metadata"),
                    UnifiedDocument.metadata_.label("document_metadata"),
                    UnifiedDocument.status.label("document_status"),
                    UnifiedDocument.extraction_incomplete,
                    UnifiedChunk.quality_warnings,
                    UnifiedChunk.graphiti_verified,
                    UnifiedChunk.graph_indexed_at,
                )
                .join_from(
                    UnifiedChunk,
                    UnifiedDocument,
                    UnifiedDocument.id == UnifiedChunk.document_id,
                )
                .where(UnifiedChunk.id == uuid_any("chunk_ids", chunk_ids))
            )
            if tenant_id is not None:
                statement = statement.where(UnifiedChunk.tenant_id == tenant_id)
            if allowed_doc_ids is not None:
                # Fail closed: only ``None`` omits the allowlist.  An empty
                # allowlist is an explicit empty scope and must match no
                # documents — ``x = ANY('{}')`` is FALSE in PostgreSQL, so
                # passing the empty sequence through keeps the predicate and
                # every row out.
                statement = statement.where(
                    UnifiedChunk.document_id == uuid_any("allowed_doc_ids", allowed_doc_ids)
                )
            if exact_phrase:
                statement = statement.where(
                    UnifiedChunk.search_text.ilike(_phrase_like_pattern(exact_phrase), escape="\\")
                )
            result = await self.session.execute(statement)
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
            "pipeline_version": statement.excluded.pipeline_version,
            "contract_index": statement.excluded.contract_index,
            "graph_indexed_at": statement.excluded.graph_indexed_at,
            "updated_at": statement.excluded.updated_at,
            "instrument_name": statement.excluded.instrument_name,
            "section_ref": statement.excluded.section_ref,
            "instrument_year": statement.excluded.instrument_year,
        },
    )


def build_chunk_rows(
    *,
    document_id: str,
    user_id: str,
    chunks: Sequence[dict[str, Any]],
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    tenant_id = _resolve_tenant_id(tenant_id, user_id)
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
                "tenant_id": tenant_id,
                "document_version": int(chunk.get("document_version", 1)),
                "contract_index": (
                    int(chunk["contract_index"])
                    if isinstance(chunk.get("contract_index"), int)
                    and not isinstance(chunk.get("contract_index"), bool)
                    and int(chunk["contract_index"]) >= 0
                    else None
                ),
                "pipeline_version": str(
                    chunk.get("pipeline_version") or CANONICAL_PIPELINE_VERSION
                ),
                "graph_indexed_at": (
                    chunk.get("graph_indexed_at")
                    or (now if bool(chunk.get("graphiti_verified", False)) else None)
                ),
                "locus": locus,
                # The upsert's DO UPDATE SET refreshes updated_at from the row, so
                # every write path carries it; callers may pin their own value.
                "updated_at": chunk.get("updated_at", now),
            }
        )
    return rows


def build_search_filter_params(*, metadata_filter: dict[str, Any]) -> dict[str, Any]:
    raw_document_ids = metadata_filter.get("document_ids")
    document_ids = list(raw_document_ids) if raw_document_ids is not None else []
    parties = metadata_filter.get("parties") or []
    chunk_ids = metadata_filter.get("chunk_ids") or []
    return {
        "document_ids": document_ids,
        "scope_bound": bool(metadata_filter.get("scope_bound", raw_document_ids is not None)),
        "chunk_ids": chunk_ids,
        "document_kind": metadata_filter.get("document_kind"),
        "jurisdiction": metadata_filter.get("jurisdiction"),
        "contract_type": metadata_filter.get("contract_type"),
        "contract_index": metadata_filter.get("contract_index"),
        "clause_type": metadata_filter.get("clause_type"),
        "require_graphiti_verified": bool(metadata_filter.get("require_graphiti_verified")),
        "tenant_id": metadata_filter.get("tenant_id"),
        "metadata_filter": json.dumps(metadata_filter.get("metadata_", {})),
        "parties_filter": json.dumps(parties),
    }


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{value:.12f}" for value in embedding) + "]"
