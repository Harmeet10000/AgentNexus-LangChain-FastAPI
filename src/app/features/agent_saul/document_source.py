"""Canonical document-content source behind one injected Protocol.

Discovery (2026-10): a real WS turn reached the ingestion node whose body fed
normalisation from a hardcoded ``text: str = ""`` stub, so every turn ended
``outcome=failed`` with ``MISSING_DOCUMENT_TEXT`` /
``MISSING_NORMALIZED_DOCUMENT``. The text is not a column on ``UnifiedDocument``
(that row stores ``object_uri``, the object-storage pointer): the Postgres
retrieval truth is ``UnifiedChunk`` — its ``content`` rows, unique per
``(document_id, document_version, chunk_index)``, reassemble the document text
in chunk order.

The agent_saul feature owns the Protocol; the concrete adapter reaches through
the documents feature's repository boundary (the feature→feature edge is
established — ``agent_saul.dependencies`` already binds auth, and no
import-linter contract forbids it). Injection rides the session's
``RunnableConfig`` (``configurable``): the Postgres checkpointer repopulates
the persisted configurable from ``thread_id``/``checkpoint_ns``/``checkpoint_id``
only (``langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.aput``), so this
per-session object never reaches a checkpoint blob while every resume round
keeps it — the graph itself stays compiled once per process.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.features.documents.exceptions import DocumentNotFoundError
from app.features.documents.models import UnifiedChunk
from app.features.documents.repository import DocumentRepository
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Final

    from langchain_core.runnables import RunnableConfig
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from sqlalchemy.sql.selectable import Select

    from app.features.documents.exceptions import DocumentDatabaseError

# ---------------------------------------------------------------------------
# Failure taxonomy (caller-side union, translated into feature unions at edges)
# ---------------------------------------------------------------------------


class DocumentTextFailureCode(StrEnum):
    """Why a canonical document-content read produced no text."""

    NOT_FOUND = "not_found"
    EMPTY = "empty"
    UNAVAILABLE = "unavailable"


class DocumentTextFailure(BaseModel):
    """Typed outcome of a failed document-content read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: DocumentTextFailureCode
    message: str
    doc_id: str
    details: dict[str, object] | None = None


type DocumentContentResult[T] = Result[T, DocumentTextFailure]


DocumentReadinessState = Literal[
    "completed",
    "completed_with_warnings",
    "not_ready",
    "failed",
]


class DocumentReadiness(BaseModel, frozen=True):
    """Persisted ingestion state used by Saul's admission gate.

    ``completed_with_warnings`` is deliberately a separate terminal state.  A
    caller may choose to accept it, but it must never be silently collapsed
    into either ``completed`` or ``no_evidence``.
    """

    status: DocumentReadinessState
    warning_count: int = 0
    warnings: tuple[dict[str, object], ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False


class CanonicalDocumentChunk(BaseModel, frozen=True):
    """One persisted chunk, with enough identity to cite it safely.

    The Saul graph must never turn a set of documents into one anonymous text
    string.  ``document_id`` + ``document_version`` + ``chunk_index`` is the
    canonical identity even when a test double does not provide the database
    UUID in ``chunk_id``.
    """

    chunk_id: str
    document_id: str
    document_version: int
    chunk_index: int
    # Stable per-file contract identity. ``None`` is legacy/unknown.
    contract_index: int | None = None
    content: str
    page_no: int = 0
    chunk_kind: str = "generic"
    locus: str | None = None
    clause_type: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True)


def _document_not_found(*, user_id: str, doc_id: str, reason: str) -> DocumentTextFailure:
    return DocumentTextFailure(
        code=DocumentTextFailureCode.NOT_FOUND,
        message=reason,
        doc_id=doc_id,
        details={"user_id": user_id, "source": "document_content_source"},
    )


def _document_unavailable(*, user_id: str, doc_id: str) -> DocumentTextFailure:
    return DocumentTextFailure(
        code=DocumentTextFailureCode.UNAVAILABLE,
        message="Document content source is unavailable",
        doc_id=doc_id,
        details={"user_id": user_id, "source": "document_content_source"},
    )


def _translate_document_failure(
    error: DocumentNotFoundError | DocumentDatabaseError,
    *,
    user_id: str,
    doc_id: str,
) -> DocumentTextFailure:
    """Repo failure → caller-side terminology.

    The repository read is user-scoped, so an absent global row and a foreign
    doc_id are indistinguishable by design — refuse before revealing, the same
    owner-gate semantics the WS handshake applies to threads.
    """
    if isinstance(error, DocumentNotFoundError):
        return _document_not_found(
            user_id=user_id,
            doc_id=doc_id,
            reason=error.message,
        )
    return _document_unavailable(user_id=user_id, doc_id=doc_id)


# ---------------------------------------------------------------------------
# Injection contract
# ---------------------------------------------------------------------------


class DocumentContentSource(Protocol):
    """Document reads the Saul pipeline needs: existence+ownership and text."""

    async def ensure_document(
        self, *, user_id: str, doc_id: str, tenant_id: str | None = None
    ) -> DocumentContentResult[None]:
        """Confirm the doc exists for this user; never reads its content."""
        ...

    async def check_readiness(
        self, *, user_id: str, doc_id: str, tenant_id: str | None = None
    ) -> DocumentContentResult[DocumentReadiness]:
        """Return the persisted ingestion state for an authorized document."""
        ...

    async def fetch_text(
        self, *, user_id: str, doc_id: str, tenant_id: str | None = None
    ) -> DocumentContentResult[str]:
        """Reassemble the document's canonical text."""
        ...

    async def fetch_segments(
        self,
        *,
        user_id: str,
        doc_ids: Sequence[str],
        tenant_id: str | None = None,
    ) -> DocumentContentResult[tuple[CanonicalDocumentChunk, ...]]:
        """Read canonical chunks without crossing document boundaries."""
        ...


DOCUMENT_SOURCE_CONFIG_KEY: Final[str] = "saul_document_content_source"


def document_content_source_from_config(
    config: RunnableConfig,
) -> DocumentContentSource | None:
    """Read the per-session injected source off the invoke config.

    The only writer of this key is ``AgentSaulService._build_config``; anything
    else present is wiring debris and resolves to ``None``, which the node
    reports as a clean state error instead of silently continuing.
    """
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    source = configurable.get(DOCUMENT_SOURCE_CONFIG_KEY)
    if source is None:
        return None
    return cast("DocumentContentSource", source)


# ---------------------------------------------------------------------------
# Concrete adapter: Postgres retrieval truth (documents feature boundary)
# ---------------------------------------------------------------------------


def parse_document_id(doc_id: str) -> UUID | None:
    """Canonical row id is a UUID; anything else cannot exist."""
    try:
        return UUID(doc_id)
    except (TypeError, ValueError):
        return None


def build_latest_version_statement(
    user_id: str,
    doc_uuid: UUID,
    *,
    tenant_id: str | None = None,
) -> Select[tuple[str]]:
    """Chunk contents of the newest version, in document order.

    ``chunk_index`` is unique within ``(document_id, document_version)``
    (``uq_chunks_document_version_chunk_index``) and the chunker is
    order-preserving, so ordering by it reassembles the document.
    """
    owner_scope = (
        UnifiedChunk.tenant_id == tenant_id
        if tenant_id is not None
        else UnifiedChunk.user_id == user_id
    )
    latest_version = (
        select(func.max(UnifiedChunk.document_version))
        .where(owner_scope, UnifiedChunk.document_id == doc_uuid)
        .scalar_subquery()
    )
    owner_scope = (
        UnifiedChunk.tenant_id == tenant_id
        if tenant_id is not None
        else UnifiedChunk.user_id == user_id
    )
    return (
        select(UnifiedChunk.content)
        .where(
            owner_scope,
            UnifiedChunk.document_id == doc_uuid,
            UnifiedChunk.document_version == latest_version,
        )
        .order_by(UnifiedChunk.chunk_index.asc())
    )


async def _fetch_latest_version_contents(
    session: AsyncSession,
    *,
    user_id: str,
    doc_uuid: UUID,
    tenant_id: str | None = None,
) -> tuple[str, ...]:
    statement = build_latest_version_statement(user_id, doc_uuid, tenant_id=tenant_id)
    rows: Sequence[str] = (await session.execute(statement)).scalars().all()
    return tuple(content for content in rows if content)


def build_latest_version_chunks_statement(
    user_id: str,
    doc_ids: Sequence[UUID],
    *,
    tenant_id: str | None = None,
) -> Select[tuple[UnifiedChunk]]:
    """Select candidate chunks for the requested documents.

    Version selection is intentionally completed in Python after this bounded
    query.  That keeps the statement easy to audit and, importantly, makes the
    ordering contract explicit: document identity first, then chunk index.
    """
    scope = [UnifiedChunk.document_id.in_(list(doc_ids))]
    if tenant_id is None:
        scope.append(UnifiedChunk.user_id == user_id)
    else:
        scope.append(UnifiedChunk.tenant_id == tenant_id)
    return (
        select(UnifiedChunk)
        .where(*scope)
        .order_by(
            UnifiedChunk.document_id.asc(),
            UnifiedChunk.document_version.desc(),
            UnifiedChunk.chunk_index.asc(),
        )
    )


def _latest_chunks(rows: Sequence[UnifiedChunk]) -> tuple[UnifiedChunk, ...]:
    """Retain only the newest version per document in stable order."""
    latest: dict[UUID, int] = {}
    for row in rows:
        latest[row.document_id] = max(
            latest.get(row.document_id, row.document_version), row.document_version
        )
    return tuple(
        sorted(
            (row for row in rows if row.document_version == latest.get(row.document_id)),
            key=lambda row: (str(row.document_id), row.chunk_index, str(row.id)),
        )
    )


class PostgresDocumentContentSource:
    """Reads content through the documents feature's repository boundary.

    Text contract: ``UnifiedChunk.content`` rows of the latest document
    version, ordered by ``chunk_index``, joined with ``"\n\n"``. Chunk
    boundaries are token-count derived, never mid-word, so reassembly loses
    nothing but the artificial split points.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        repository_factory: Callable[[AsyncSession], DocumentRepository] = DocumentRepository,
    ) -> None:
        self._session_factory = session_factory
        self._repository_factory = repository_factory

    async def ensure_document(
        self, *, user_id: str, doc_id: str, tenant_id: str | None = None
    ) -> DocumentContentResult[None]:
        doc_uuid = parse_document_id(doc_id)
        if doc_uuid is None:
            return Failure(
                _document_not_found(
                    user_id=user_id,
                    doc_id=doc_id,
                    reason="Requested document id is not a valid identifier",
                )
            )
        try:
            async with self._session_factory() as session:
                presence = await self._probe_document(
                    session, user_id=user_id, doc_uuid=doc_uuid, doc_id=doc_id, tenant_id=tenant_id
                )
        except SQLAlchemyError as exc:
            exc.add_note(f"doc_id={doc_id}, user_id={user_id}, operation=ensure_document")
            logger.bind(operation="ensure_document", doc_id=doc_id, user_id=user_id).exception(
                "Document content source probe failed"
            )
            return Failure(_document_unavailable(user_id=user_id, doc_id=doc_id))
        if isinstance(presence, Failure):
            return presence
        return Success(None)

    async def check_readiness(  # noqa: PLR0914 — status mapping keeps all readiness evidence together
        self, *, user_id: str, doc_id: str, tenant_id: str | None = None
    ) -> DocumentContentResult[DocumentReadiness]:
        """Read status only after proving tenant ownership.

        The readiness check intentionally uses the repository status query
        rather than inferring readiness from chunk presence.  A document can
        have partial chunks while its worker is still running or while a
        permanent failure has already been recorded.
        """
        doc_uuid = parse_document_id(doc_id)
        if doc_uuid is None:
            return Failure(
                _document_not_found(
                    user_id=user_id,
                    doc_id=doc_id,
                    reason="Requested document id is not a valid identifier",
                )
            )
        try:
            async with self._session_factory() as session:
                presence = await self._probe_document(
                    session,
                    user_id=user_id,
                    doc_uuid=doc_uuid,
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                )
                if isinstance(presence, Failure):
                    return presence
                repository = self._repository_factory(session)
                status_result = await repository.fetch_status(
                    user_id=user_id,
                    document_id=doc_id,
                    tenant_id=tenant_id,
                )
        except SQLAlchemyError as exc:
            exc.add_note(f"doc_id={doc_id}, user_id={user_id}, operation=check_readiness")
            logger.bind(operation="check_readiness", doc_id=doc_id, user_id=user_id).exception(
                "Document readiness probe failed"
            )
            return Failure(_document_unavailable(user_id=user_id, doc_id=doc_id))
        if isinstance(status_result, Failure):
            return Failure(_document_unavailable(user_id=user_id, doc_id=doc_id))
        record = status_result.unwrap()
        if record is None:
            return Failure(
                _document_not_found(user_id=user_id, doc_id=doc_id, reason="Document not found")
            )
        raw_status = str(record.get("status") or "received").lower()
        warnings_raw = record.get("warnings")
        warnings: list[dict[str, object]] = []
        if isinstance(warnings_raw, list):
            warnings = [item for item in warnings_raw if isinstance(item, dict)]
        extraction_incomplete = bool(record.get("extraction_incomplete"))
        if raw_status in {"completed", "completed_with_warnings"}:
            state: DocumentReadinessState = (
                "completed_with_warnings"
                if raw_status == "completed_with_warnings" or extraction_incomplete
                else "completed"
            )
        elif raw_status in {"failed", "error", "rejected"}:
            state = "failed"
        else:
            state = "not_ready"
        metadata = record.get("document_metadata")
        error_code = None
        error_message = None
        retryable = False
        if isinstance(metadata, dict):
            error = metadata.get("ingestion_error")
            if isinstance(error, dict):
                raw_code = error.get("code")
                raw_message = error.get("message")
                error_code = raw_code if isinstance(raw_code, str) else None
                error_message = raw_message if isinstance(raw_message, str) else None
                retryable = bool(error.get("retryable"))
        return Success(
            DocumentReadiness(
                status=state,
                warning_count=len(warnings),
                warnings=tuple(warnings),
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
            )
        )

    async def fetch_text(
        self, *, user_id: str, doc_id: str, tenant_id: str | None = None
    ) -> DocumentContentResult[str]:
        doc_uuid = parse_document_id(doc_id)
        if doc_uuid is None:
            return Failure(
                _document_not_found(
                    user_id=user_id,
                    doc_id=doc_id,
                    reason="Requested document id is not a valid identifier",
                )
            )
        try:
            async with self._session_factory() as session:
                presence = await self._probe_document(
                    session, user_id=user_id, doc_uuid=doc_uuid, doc_id=doc_id, tenant_id=tenant_id
                )
                if isinstance(presence, Failure):
                    return presence
                contents = await _fetch_latest_version_contents(
                    session, user_id=user_id, doc_uuid=doc_uuid, tenant_id=tenant_id
                )
        except SQLAlchemyError as exc:
            exc.add_note(f"doc_id={doc_id}, user_id={user_id}, operation=fetch_document_text")
            logger.bind(operation="fetch_document_text", doc_id=doc_id, user_id=user_id).exception(
                "Document content source read failed"
            )
            return Failure(_document_unavailable(user_id=user_id, doc_id=doc_id))
        text = "\n\n".join(contents)
        if not text:
            return Failure(
                DocumentTextFailure(
                    code=DocumentTextFailureCode.EMPTY,
                    message="Document exists but has no retrievable content",
                    doc_id=doc_id,
                    details={"user_id": user_id, "source": "document_content_source"},
                )
            )
        return Success(text)

    async def fetch_segments(
        self,
        *,
        user_id: str,
        doc_ids: Sequence[str],
        tenant_id: str | None = None,
    ) -> DocumentContentResult[tuple[CanonicalDocumentChunk, ...]]:
        """Fetch all selected documents as canonical, boundary-preserving chunks."""
        unique_ids = tuple(sorted(dict.fromkeys(doc_ids)))
        if not unique_ids:
            return Failure(
                DocumentTextFailure(
                    code=DocumentTextFailureCode.EMPTY,
                    message="No documents were selected for analysis",
                    doc_id="",
                    details={"user_id": user_id, "source": "document_content_source"},
                )
            )
        parsed_ids = [(doc_id, parse_document_id(doc_id)) for doc_id in unique_ids]
        invalid = next((doc_id for doc_id, parsed in parsed_ids if parsed is None), None)
        if invalid is not None:
            return Failure(
                _document_not_found(
                    user_id=user_id,
                    doc_id=invalid,
                    reason="Requested document id is not a valid identifier",
                )
            )
        uuids = [cast("UUID", parsed) for _, parsed in parsed_ids]
        try:
            async with self._session_factory() as session:
                for doc_id, doc_uuid in parsed_ids:
                    presence = await self._probe_document(
                        session,
                        user_id=user_id,
                        doc_uuid=cast("UUID", doc_uuid),
                        doc_id=doc_id,
                        tenant_id=tenant_id,
                    )
                    if isinstance(presence, Failure):
                        return presence
                result = await session.execute(
                    build_latest_version_chunks_statement(
                        user_id,
                        uuids,
                        tenant_id=tenant_id,
                    )
                )
                rows = tuple(result.scalars().all())
        except SQLAlchemyError as exc:
            exc.add_note(
                f"doc_ids={','.join(unique_ids)}, user_id={user_id}, operation=fetch_document_segments"
            )
            logger.bind(
                operation="fetch_document_segments", doc_ids=unique_ids, user_id=user_id
            ).exception("Document content source read failed")
            return Failure(_document_unavailable(user_id=user_id, doc_id=unique_ids[0]))

        chunks = _latest_chunks(rows)
        if not chunks:
            return Failure(
                DocumentTextFailure(
                    code=DocumentTextFailureCode.EMPTY,
                    message="Selected documents have no retrievable canonical chunks",
                    doc_id=unique_ids[0],
                    details={
                        "user_id": user_id,
                        "source": "document_content_source",
                        "document_ids": list(unique_ids),
                    },
                )
            )
        expected = {cast("UUID", parsed) for _, parsed in parsed_ids}
        found = {chunk.document_id for chunk in chunks}
        missing = expected - found
        if missing:
            return Failure(
                DocumentTextFailure(
                    code=DocumentTextFailureCode.EMPTY,
                    message="At least one selected document has no retrievable canonical chunks",
                    doc_id=next(doc_id for doc_id, parsed in parsed_ids if parsed in missing),
                    details={
                        "user_id": user_id,
                        "source": "document_content_source",
                        "missing_document_ids": [str(value) for value in sorted(missing, key=str)],
                    },
                )
            )
        canonical_chunks = tuple(
            CanonicalDocumentChunk(
                chunk_id=str(chunk.id),
                document_id=str(chunk.document_id),
                document_version=chunk.document_version,
                chunk_index=chunk.chunk_index,
                contract_index=chunk.contract_index,
                content=chunk.content,
                page_no=chunk.page_no,
                chunk_kind=chunk.chunk_kind,
                locus=chunk.locus,
                clause_type=chunk.clause_type,
                metadata={**chunk.metadata_, **chunk.custom_metadata},
            )
            for chunk in chunks
            if chunk.content.strip()
        )
        if not canonical_chunks:
            return Failure(
                DocumentTextFailure(
                    code=DocumentTextFailureCode.EMPTY,
                    message="Selected documents have no non-empty canonical chunks",
                    doc_id=unique_ids[0],
                    details={"user_id": user_id, "source": "document_content_source"},
                )
            )
        return Success(canonical_chunks)

    async def fetch_canonical_chunks(
        self,
        *,
        user_id: str,
        doc_ids: Sequence[str],
        tenant_id: str | None = None,
    ) -> DocumentContentResult[tuple[CanonicalDocumentChunk, ...]]:
        """Compatibility alias for callers that use the storage terminology."""
        return await self.fetch_segments(user_id=user_id, doc_ids=doc_ids, tenant_id=tenant_id)

    async def fetch_chunks(
        self,
        *,
        user_id: str,
        doc_ids: Sequence[str],
        tenant_id: str | None = None,
    ) -> DocumentContentResult[tuple[CanonicalDocumentChunk, ...]]:
        """Compatibility alias for generic chunk-source consumers."""
        return await self.fetch_segments(user_id=user_id, doc_ids=doc_ids, tenant_id=tenant_id)

    async def _probe_document(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        doc_uuid: UUID,
        doc_id: str,
        tenant_id: str | None = None,
    ) -> DocumentContentResult[None]:
        """Existence+ownership through the documents repository (user-scoped)."""
        repository = self._repository_factory(session)
        kwargs: dict[str, str] = {"user_id": user_id, "document_id": str(doc_uuid)}
        if tenant_id is not None:
            kwargs["tenant_id"] = tenant_id
        result = await repository.get_document_by_id(**kwargs)
        if isinstance(result, Failure):
            return Failure(
                _translate_document_failure(
                    result.failure(),
                    user_id=user_id,
                    doc_id=doc_id,
                )
            )
        return Success(None)


def postgres_document_content_source(
    session_factory: async_sessionmaker[AsyncSession],
) -> PostgresDocumentContentSource:
    """Wiring site: build the concrete source for one runtime session."""
    return PostgresDocumentContentSource(session_factory)
