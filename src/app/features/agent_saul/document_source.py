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
from typing import TYPE_CHECKING, Protocol, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict
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

    async def ensure_document(self, *, user_id: str, doc_id: str) -> DocumentContentResult[None]:
        """Confirm the doc exists for this user; never reads its content."""
        ...

    async def fetch_text(self, *, user_id: str, doc_id: str) -> DocumentContentResult[str]:
        """Reassemble the document's canonical text."""
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


def build_latest_version_statement(user_id: str, doc_uuid: UUID) -> Select[tuple[str]]:
    """Chunk contents of the newest version, in document order.

    ``chunk_index`` is unique within ``(document_id, document_version)``
    (``uq_chunks_document_version_chunk_index``) and the chunker is
    order-preserving, so ordering by it reassembles the document.
    """
    latest_version = (
        select(func.max(UnifiedChunk.document_version))
        .where(
            UnifiedChunk.user_id == user_id,
            UnifiedChunk.document_id == doc_uuid,
        )
        .scalar_subquery()
    )
    return (
        select(UnifiedChunk.content)
        .where(
            UnifiedChunk.user_id == user_id,
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
) -> tuple[str, ...]:
    statement = build_latest_version_statement(user_id, doc_uuid)
    rows: Sequence[str] = (await session.execute(statement)).scalars().all()
    return tuple(content for content in rows if content)


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

    async def ensure_document(self, *, user_id: str, doc_id: str) -> DocumentContentResult[None]:
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
                    session, user_id=user_id, doc_uuid=doc_uuid, doc_id=doc_id
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

    async def fetch_text(self, *, user_id: str, doc_id: str) -> DocumentContentResult[str]:
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
                    session, user_id=user_id, doc_uuid=doc_uuid, doc_id=doc_id
                )
                if isinstance(presence, Failure):
                    return presence
                contents = await _fetch_latest_version_contents(
                    session, user_id=user_id, doc_uuid=doc_uuid
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

    async def _probe_document(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        doc_uuid: UUID,
        doc_id: str,
    ) -> DocumentContentResult[None]:
        """Existence+ownership through the documents repository (user-scoped)."""
        repository = self._repository_factory(session)
        result = await repository.get_document_by_id(user_id=user_id, document_id=str(doc_uuid))
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
