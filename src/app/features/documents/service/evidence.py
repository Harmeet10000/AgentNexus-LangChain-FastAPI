"""Scoped, handle-based evidence access for Agent Saul and other callers.

This module intentionally does not replace the legacy search endpoint.  The
legacy endpoint returns hydrated chunks for compatibility; this API is the
smaller seam used by agents: retrieve candidates, issue opaque handles, and
dereference only after re-authorizing the current principal and version.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, cast

from pydantic import BaseModel, ConfigDict
from returns.result import Failure

from app.ai.embeddings import EmbeddingTaskType, embed_text
from app.features.documents.access_audit import (
    DocumentAccessAuditEvent,
    DocumentAccessAuditRecorder,
    InMemoryDocumentAccessAuditRecorder,
    SqlDocumentAccessAuditRecorder,
)
from app.features.documents.dto import (
    EvidenceCoverage,
    EvidencePolicy,
    EvidenceReadItem,
    EvidenceReadResponse,
    EvidenceRef,
    EvidenceSearchItem,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
    IngestionDocumentState,
    IngestionStateResponse,
)
from app.features.documents.exceptions import DocumentStatusNotFoundError
from app.features.documents.repository import DocumentRepository, build_search_filter_params
from app.features.documents.service.query import retrieve_fused
from app.observability.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# At least the 24-hour HITL pause window; configuration may extend this.
DEFAULT_HANDLE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_SNIPPET_CHARS = 800


class EvidenceHandle(BaseModel):
    """Only metadata needed to authorize a dereference; never chunk content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    chunk_id: str
    tenant_id: str
    principal_id: str
    thread_id: str
    run_id: str
    # Every handle is bound to a checkpoint generation.  The deterministic
    # run marker is used only by legacy callers that do not expose a concrete
    # LangGraph checkpoint; it is still scoped to this run and never read from
    # model state.
    checkpoint_id: str
    allowed_doc_ids: tuple[str, ...] = ()
    document_id: str
    document_version: int
    contract_index: int | None = None
    pipeline_version: str | None = None
    expires_at: datetime


class EvidenceHandleStore(Protocol):
    async def issue(self, handle: EvidenceHandle, *, ttl_seconds: int) -> None: ...

    async def resolve(self, evidence_id: str) -> EvidenceHandle | None: ...

    async def was_expired(self, evidence_id: str) -> bool: ...


class InMemoryEvidenceHandleStore:
    """Test/local store; production wiring should use ``RedisEvidenceHandleStore``."""

    def __init__(self) -> None:
        self._handles: dict[str, EvidenceHandle] = {}
        self._expired: set[str] = set()
        self._lock = asyncio.Lock()

    async def issue(self, handle: EvidenceHandle, *, ttl_seconds: int) -> None:
        del ttl_seconds  # expiry is already materialized in the scoped handle
        async with self._lock:
            self._handles[handle.evidence_id] = handle
            self._expired.discard(handle.evidence_id)

    async def resolve(self, evidence_id: str) -> EvidenceHandle | None:
        async with self._lock:
            handle = self._handles.get(evidence_id)
            if handle is None:
                return None
            if handle.expires_at <= datetime.now(tz=UTC):
                self._handles.pop(evidence_id, None)
                self._expired.add(evidence_id)
                return None
            return handle

    async def was_expired(self, evidence_id: str) -> bool:
        async with self._lock:
            return evidence_id in self._expired


class RedisEvidenceHandleStore:
    """Redis-backed handle store with TTL and no full source content."""

    def __init__(
        self,
        redis: Redis,
        *,
        prefix: str = "documents:evidence:handle:",
    ) -> None:
        self.redis = redis
        self.prefix = prefix

    async def issue(self, handle: EvidenceHandle, *, ttl_seconds: int) -> None:
        await self.redis.setex(
            f"{self.prefix}{handle.evidence_id}",
            ttl_seconds,
            handle.model_dump_json(),
        )

    async def resolve(self, evidence_id: str) -> EvidenceHandle | None:
        raw = await self.redis.get(f"{self.prefix}{evidence_id}")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return EvidenceHandle.model_validate_json(raw)

    @staticmethod
    async def was_expired(evidence_id: str) -> bool:
        # Redis expiry is intentionally indistinguishable from an unknown handle
        # to callers; the public response remains non-disclosing.
        del evidence_id
        return False


class SessionFactoryEvidenceRepository:
    """Open a short-lived repository session for each evidence operation.

    Agent Saul's graph is process-scoped and compiled once at startup, so it
    must never retain an ``AsyncSession`` inside the long-lived evidence
    service.  This adapter preserves the repository's Result contract while
    obtaining a fresh session for every retrieval/readiness call.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        async with self._session_factory() as session:
            repository = DocumentRepository(session)
            return await getattr(repository, method)(*args, **kwargs)

    async def bm25_search(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("bm25_search", *args, **kwargs)

    async def vector_search(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("vector_search", *args, **kwargs)

    async def trigram_search(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("trigram_search", *args, **kwargs)

    async def fetch_chunks_by_ids(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("fetch_chunks_by_ids", *args, **kwargs)

    async def fetch_status(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call("fetch_status", *args, **kwargs)


class EvidenceService:
    """Document-side evidence contract over deterministic fused retrieval."""

    def __init__(
        self,
        *,
        repo: Any,
        handle_store: EvidenceHandleStore | None = None,
        audit_recorder: DocumentAccessAuditRecorder | None = None,
        handle_ttl_seconds: int = DEFAULT_HANDLE_TTL_SECONDS,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    ) -> None:
        self.repo = repo
        self.handle_store = handle_store or InMemoryEvidenceHandleStore()
        self.handle_ttl_seconds = handle_ttl_seconds
        self.snippet_chars = snippet_chars
        self.audit_recorder = audit_recorder or InMemoryDocumentAccessAuditRecorder()

    async def _record_access(
        self,
        response: EvidenceSearchResponse | EvidenceReadResponse | IngestionStateResponse,
        *,
        operation: str,
        principal_id: str,
        tenant_id: str,
        requested_document_ids: Sequence[str],
        document_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        thread_id: str = "",
        run_id: str = "",
        checkpoint_id: str = "",
        evidence_policy: EvidencePolicy,
    ) -> EvidenceSearchResponse | EvidenceReadResponse | IngestionStateResponse:
        try:
            await self.audit_recorder.append(
                DocumentAccessAuditEvent(
                    tenant_id=tenant_id,
                    principal_id=principal_id,
                    thread_id=thread_id,
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    operation=operation,
                    requested_document_ids=tuple(requested_document_ids),
                    document_ids=tuple(document_ids),
                    evidence_ids=tuple(evidence_ids),
                    policy=evidence_policy,
                    outcome=response.status,
                    reason=response.message,
                )
            )
        except Exception as exc:  # noqa: BLE001 — audit failure is a fail-closed boundary
            exc.add_note(f"operation={operation}, tenant_id={tenant_id}")
            logger.bind(operation=operation, tenant_id=tenant_id).exception(
                "document_access_audit_failed"
            )
            if isinstance(response, EvidenceSearchResponse):
                return response.model_copy(
                    update={
                        "status": "transient_failure",
                        "message": "Document access audit is unavailable.",
                        "items": [],
                    }
                )
            if isinstance(response, EvidenceReadResponse):
                return response.model_copy(
                    update={
                        "status": "transient_failure",
                        "message": "Document access audit is unavailable.",
                        "items": [],
                    }
                )
            return response.model_copy(
                update={
                    "status": "transient_failure",
                    "message": "Document access audit is unavailable.",
                    "documents": [],
                }
            )
        return response

    async def search_evidence(
        self,
        *,
        principal_id: str,
        tenant_id: str,
        allowed_doc_ids: Sequence[str],
        thread_id: str,
        run_id: str,
        request: EvidenceSearchRequest,
        checkpoint_id: str | None = None,
        request_id: str | None = None,
        evidence_policy: EvidencePolicy = "mechanically_verified",
    ) -> EvidenceSearchResponse:
        response = await self._search_evidence(
            principal_id=principal_id,
            tenant_id=tenant_id,
            allowed_doc_ids=allowed_doc_ids,
            thread_id=thread_id,
            run_id=run_id,
            request=request,
            checkpoint_id=checkpoint_id,
            request_id=request_id,
            evidence_policy=evidence_policy,
        )
        return cast(
            "EvidenceSearchResponse",
            await self._record_access(
                response,
                operation="evidence.search",
                principal_id=principal_id,
                tenant_id=tenant_id,
                requested_document_ids=allowed_doc_ids,
                document_ids=allowed_doc_ids,
                thread_id=thread_id,
                run_id=run_id,
                checkpoint_id=checkpoint_id or f"run:{run_id}",
                evidence_policy=evidence_policy,
            ),
        )

    async def _search_evidence(  # noqa: PLR0914 — evidence issue, retrieval, and scope metadata
        self,
        *,
        principal_id: str,
        tenant_id: str,
        allowed_doc_ids: Sequence[str],
        thread_id: str,
        run_id: str,
        request: EvidenceSearchRequest,
        checkpoint_id: str | None = None,
        request_id: str | None = None,
        evidence_policy: EvidencePolicy = "mechanically_verified",
    ) -> EvidenceSearchResponse:
        checkpoint_id = checkpoint_id or f"run:{run_id}"
        scope = _normalized_scope(allowed_doc_ids)
        if not scope:
            return EvidenceSearchResponse(
                status="unauthorized",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message="No authorized documents were supplied for this analysis.",
            )

        # Readiness is part of the evidence contract, not an inference from
        # retrieval returning zero rows.  Without this gate a queued/failed
        # document is indistinguishable from a valid negative answer and a
        # caller could incorrectly report that the corpus contains no support.
        readiness = await self._scope_readiness(
            principal_id=principal_id,
            tenant_id=tenant_id,
            document_ids=scope,
            request_id=request_id,
            evidence_policy=evidence_policy,
        )
        if isinstance(readiness, EvidenceSearchResponse):
            return readiness

        query_embedding = await embed_text(request.query, task_type=EmbeddingTaskType.QUERY)
        filter_params = build_search_filter_params(
            metadata_filter={
                "document_ids": list(scope),
                "document_kind": request.document_kind,
                "jurisdiction": request.jurisdiction,
                "contract_type": request.contract_type,
                "contract_index": request.contract_index,
                "clause_type": request.clause_type,
                "require_graphiti_verified": request.require_graph_indexed,
            }
        )
        fused_result = await retrieve_fused(
            repo=self.repo,
            user_id=principal_id,
            tenant_id=tenant_id,
            query_text=request.query,
            query_embedding=query_embedding,
            candidate_limit=request.candidate_limit,
            limit=request.limit,
            filter_params=filter_params,
            weights=None,
        )
        if isinstance(fused_result, Failure):
            return EvidenceSearchResponse(
                status="transient_failure",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message=fused_result.failure().message,
            )

        fused, lookup = fused_result.unwrap()
        if not fused:
            return EvidenceSearchResponse(
                status="no_evidence",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message="No evidence was found in the authorized documents.",
                coverage=EvidenceCoverage(documents_requested=len(scope)),
            )

        items: list[EvidenceSearchItem] = []
        graph_indexed_doc_ids: set[str] = set()
        result_doc_ids: set[str] = set()
        for ranked in fused:
            row = lookup.get(ranked.chunk_id)
            if row is None:
                continue
            document_id = str(row.get("document_id", ""))
            # Defense in depth: the repository filter is authoritative, but do
            # not issue a capability if a narrow test double violated it.
            if document_id not in scope:
                continue
            record = _record_metadata(row)
            graph_indexed_flag = bool(row.get("graphiti_verified", False))
            result_doc_ids.add(document_id)
            if graph_indexed_flag:
                graph_indexed_doc_ids.add(document_id)
            evidence_id = f"evh_{secrets.token_urlsafe(32)}"
            handle = EvidenceHandle(
                evidence_id=evidence_id,
                chunk_id=ranked.chunk_id,
                tenant_id=tenant_id,
                principal_id=principal_id,
                thread_id=thread_id,
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                allowed_doc_ids=scope,
                document_id=document_id,
                document_version=_document_version(row),
                contract_index=_contract_index(row),
                pipeline_version=_pipeline_version(row),
                expires_at=datetime.now(tz=UTC) + timedelta(seconds=self.handle_ttl_seconds),
            )
            await self.handle_store.issue(handle, ttl_seconds=self.handle_ttl_seconds)
            ref = _evidence_ref(
                handle=handle,
                graph_indexed=graph_indexed_flag,
                score=ranked.score,
                rank=ranked.rank,
                page_no=int(row.get("page_no") or 0),
                graph_indexed_at=_graph_indexed_at(row),
            )
            content = str(row.get("content") or "")
            items.append(
                EvidenceSearchItem(
                    evidence=ref,
                    document_title=str(row.get("title") or ""),
                    snippet=_bounded_snippet(content, self.snippet_chars),
                    chunk_index=int(row.get("chunk_index") or 0),
                    contract_index=_contract_index(row),
                    chunk_kind=str(row.get("chunk_kind") or "generic"),
                    clause_type=(
                        str(row["clause_type"]) if row.get("clause_type") is not None else None
                    ),
                    metadata=record,
                )
            )

        if not items:
            return EvidenceSearchResponse(
                status="no_evidence",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message="No evidence remained after authorization filtering.",
            )
        return EvidenceSearchResponse(
            status="ok",
            evidence_policy=evidence_policy,
            request_id=request_id,
            items=items,
            coverage=EvidenceCoverage(
                documents_requested=len(scope),
                # Readiness is measured from the authorized status query, not
                # from whichever documents happened to rank into this result.
                documents_ready=sum(
                    item.state in {"ready", "completed_with_warnings"} for item in readiness
                ),
                documents_with_warnings=sum(
                    item.state == "completed_with_warnings" for item in readiness
                ),
                graph_indexed_documents=len(graph_indexed_doc_ids),
                graph_unavailable_documents=len(result_doc_ids - graph_indexed_doc_ids),
            ),
        )

    async def _scope_readiness(
        self,
        *,
        principal_id: str,
        tenant_id: str,
        document_ids: Sequence[str],
        request_id: str | None,
        evidence_policy: EvidencePolicy,
    ) -> list[IngestionDocumentState] | EvidenceSearchResponse:
        """Return status for every scoped document before searching chunks."""
        states: list[IngestionDocumentState] = []
        for document_id in document_ids:
            result = await self.repo.fetch_status(
                user_id=principal_id,
                tenant_id=tenant_id,
                document_id=document_id,
            )
            if isinstance(result, Failure):
                if isinstance(result.failure(), DocumentStatusNotFoundError):
                    return EvidenceSearchResponse(
                        status="unauthorized",
                        evidence_policy=evidence_policy,
                        request_id=request_id,
                        message="A requested document is outside the authorized corpus.",
                    )
                return EvidenceSearchResponse(
                    status="transient_failure",
                    evidence_policy=evidence_policy,
                    request_id=request_id,
                    message=result.failure().message,
                )
            record = result.unwrap()
            if record is None:
                return EvidenceSearchResponse(
                    status="unauthorized",
                    evidence_policy=evidence_policy,
                    request_id=request_id,
                )
            states.append(_ingestion_state(record, document_id))

        coverage = EvidenceCoverage(
            documents_requested=len(states),
            documents_ready=sum(item.state == "ready" for item in states),
            documents_with_warnings=sum(item.state == "completed_with_warnings" for item in states),
            documents_failed=sum(item.state == "failed" for item in states),
            documents_not_ready=sum(item.state == "index_not_ready" for item in states),
        )
        if coverage.documents_failed:
            return EvidenceSearchResponse(
                status="ingestion_failed",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message="One or more authorized documents failed ingestion.",
                coverage=coverage,
            )
        if coverage.documents_not_ready:
            return EvidenceSearchResponse(
                status="index_not_ready",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message="One or more authorized documents are not ready for evidence search.",
                coverage=coverage,
            )
        return states

    async def read_evidence_by_id(
        self,
        *,
        evidence_id: str,
        principal_id: str,
        tenant_id: str,
        allowed_doc_ids: Sequence[str],
        thread_id: str,
        run_id: str,
        checkpoint_id: str | None = None,
        request_id: str | None = None,
        evidence_policy: EvidencePolicy = "mechanically_verified",
    ) -> EvidenceReadResponse:
        response = await self._read_evidence_by_id(
            evidence_id=evidence_id,
            principal_id=principal_id,
            tenant_id=tenant_id,
            allowed_doc_ids=allowed_doc_ids,
            thread_id=thread_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            request_id=request_id,
            evidence_policy=evidence_policy,
        )
        return cast(
            "EvidenceReadResponse",
            await self._record_access(
                response,
                operation="evidence.read",
                principal_id=principal_id,
                tenant_id=tenant_id,
                requested_document_ids=allowed_doc_ids,
                document_ids=((response.items[0].evidence.document_id,) if response.items else ()),
                evidence_ids=(evidence_id,),
                thread_id=thread_id,
                run_id=run_id,
                checkpoint_id=checkpoint_id or f"run:{run_id}",
                evidence_policy=evidence_policy,
            ),
        )

    async def _read_evidence_by_id(
        self,
        *,
        evidence_id: str,
        principal_id: str,
        tenant_id: str,
        allowed_doc_ids: Sequence[str],
        thread_id: str,
        run_id: str,
        checkpoint_id: str | None = None,
        request_id: str | None = None,
        evidence_policy: EvidencePolicy = "mechanically_verified",
    ) -> EvidenceReadResponse:
        checkpoint_id = checkpoint_id or f"run:{run_id}"
        handle = await self.handle_store.resolve(evidence_id)
        if handle is None:
            status = (
                "expired" if await self.handle_store.was_expired(evidence_id) else "invalid_handle"
            )
            return EvidenceReadResponse(
                status=status,
                evidence_policy=evidence_policy,
                request_id=request_id,
                message="Evidence handle is unavailable.",
            )
        if handle.expires_at <= datetime.now(tz=UTC):
            return EvidenceReadResponse(
                status="expired", evidence_policy=evidence_policy, request_id=request_id
            )
        current_scope = _normalized_scope(allowed_doc_ids)
        if not _handle_matches_scope(
            handle=handle,
            principal_id=principal_id,
            tenant_id=tenant_id,
            thread_id=thread_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            current_scope=current_scope,
        ):
            return EvidenceReadResponse(
                status="unauthorized",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message="Evidence handle is outside the current authorization scope.",
            )

        chunk_result = await self.repo.fetch_chunks_by_ids(
            [handle.chunk_id],
            tenant_id=tenant_id,
            allowed_doc_ids=current_scope,
        )
        if isinstance(chunk_result, Failure):
            return EvidenceReadResponse(
                status="transient_failure",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message=chunk_result.failure().message,
            )
        row = chunk_result.unwrap().get(handle.chunk_id)
        if row is None or str(row.get("document_id", "")) != handle.document_id:
            return EvidenceReadResponse(
                status="unauthorized", evidence_policy=evidence_policy, request_id=request_id
            )
        current_version = _document_version(row)
        current_contract_index = _contract_index(row)
        current_pipeline = _pipeline_version(row)
        if (
            current_version != handle.document_version
            or current_contract_index != handle.contract_index
            or current_pipeline != handle.pipeline_version
        ):
            return EvidenceReadResponse(
                status="stale_version",
                evidence_policy=evidence_policy,
                request_id=request_id,
                message="Evidence handle refers to an older document version.",
            )
        ref = _evidence_ref(
            handle=handle,
            graph_indexed=bool(row.get("graphiti_verified", False)),
            score=0.0,
            rank=0,
            page_no=int(row.get("page_no") or 0),
            graph_indexed_at=_graph_indexed_at(row),
        )
        return EvidenceReadResponse(
            status="ok",
            evidence_policy=evidence_policy,
            request_id=request_id,
            items=[
                EvidenceReadItem(
                    evidence=ref,
                    document_title=str(row.get("title") or ""),
                    content=str(row.get("content") or ""),
                    preamble=str(row.get("preamble") or ""),
                    contract_index=_contract_index(row),
                    metadata=_record_metadata(row),
                )
            ],
            coverage=EvidenceCoverage(
                documents_requested=1,
                documents_ready=1,
                graph_indexed_documents=int(bool(row.get("graphiti_verified", False))),
                graph_unavailable_documents=int(not bool(row.get("graphiti_verified", False))),
            ),
        )

    async def get_ingestion_state(
        self,
        *,
        principal_id: str,
        tenant_id: str,
        allowed_doc_ids: Sequence[str],
        document_ids: Sequence[str] | None = None,
        request_id: str | None = None,
        thread_id: str = "",
        run_id: str = "",
        checkpoint_id: str | None = None,
        evidence_policy: EvidencePolicy = "mechanically_verified",
    ) -> IngestionStateResponse:
        response = await self._get_ingestion_state(
            principal_id=principal_id,
            tenant_id=tenant_id,
            allowed_doc_ids=allowed_doc_ids,
            document_ids=document_ids,
            request_id=request_id,
            evidence_policy=evidence_policy,
        )
        return cast(
            "IngestionStateResponse",
            await self._record_access(
                response,
                operation="document.read",
                principal_id=principal_id,
                tenant_id=tenant_id,
                requested_document_ids=document_ids or allowed_doc_ids,
                document_ids=tuple(item.document_id for item in response.documents),
                thread_id=thread_id,
                run_id=run_id,
                checkpoint_id=checkpoint_id or (f"run:{run_id}" if run_id else ""),
                evidence_policy=evidence_policy,
            ),
        )

    async def _get_ingestion_state(
        self,
        *,
        principal_id: str,
        tenant_id: str,
        allowed_doc_ids: Sequence[str],
        document_ids: Sequence[str] | None = None,
        request_id: str | None = None,
        evidence_policy: EvidencePolicy = "mechanically_verified",
    ) -> IngestionStateResponse:
        scope = _normalized_scope(allowed_doc_ids)
        requested = _normalized_scope(document_ids or scope)
        if not requested or any(document_id not in scope for document_id in requested):
            return IngestionStateResponse(
                status="unauthorized", evidence_policy=evidence_policy, request_id=request_id
            )
        states: list[IngestionDocumentState] = []
        for document_id in requested:
            result = await self.repo.fetch_status(
                user_id=principal_id, tenant_id=tenant_id, document_id=document_id
            )
            if isinstance(result, Failure):
                if isinstance(result.failure(), DocumentStatusNotFoundError):
                    return IngestionStateResponse(
                        status="unauthorized",
                        evidence_policy=evidence_policy,
                        request_id=request_id,
                    )
                return IngestionStateResponse(
                    status="transient_failure",
                    evidence_policy=evidence_policy,
                    request_id=request_id,
                    message=result.failure().message,
                )
            record = result.unwrap()
            if record is None:
                return IngestionStateResponse(
                    status="unauthorized", evidence_policy=evidence_policy, request_id=request_id
                )
            states.append(_ingestion_state(record, document_id))

        if any(item.state == "failed" for item in states):
            status = "ingestion_failed"
        elif any(item.state == "index_not_ready" for item in states):
            status = "index_not_ready"
        else:
            status = "ok"
        return IngestionStateResponse(
            status=status,
            evidence_policy=evidence_policy,
            request_id=request_id,
            documents=states,
            coverage=EvidenceCoverage(
                documents_requested=len(states),
                documents_ready=sum(item.state == "ready" for item in states),
                documents_with_warnings=sum(
                    item.state == "completed_with_warnings" for item in states
                ),
                documents_failed=sum(item.state == "failed" for item in states),
                documents_not_ready=sum(item.state == "index_not_ready" for item in states),
                graph_indexed_documents=sum(item.graph_indexed_chunk_count > 0 for item in states),
                graph_unavailable_documents=sum(
                    item.graph_indexed_chunk_count == 0 for item in states
                ),
            ),
        )


def build_evidence_service(
    *,
    repo: DocumentRepository | Any | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    redis: Redis | None = None,
    handle_ttl_seconds: int = DEFAULT_HANDLE_TTL_SECONDS,
    audit_recorder: DocumentAccessAuditRecorder | None = None,
) -> EvidenceService:
    """Construct the production evidence service with an explicit handle backend."""
    if repo is not None and session_factory is not None:
        message = "pass either repo or session_factory, not both"
        raise ValueError(message)
    if repo is None:
        if session_factory is None:
            message = "repo or session_factory is required"
            raise ValueError(message)
        repo = SessionFactoryEvidenceRepository(session_factory)
    store: EvidenceHandleStore = (
        RedisEvidenceHandleStore(redis) if redis is not None else InMemoryEvidenceHandleStore()
    )
    if audit_recorder is None and session_factory is not None:
        audit_recorder = SqlDocumentAccessAuditRecorder(session_factory)
    return EvidenceService(
        repo=repo,
        handle_store=store,
        handle_ttl_seconds=handle_ttl_seconds,
        audit_recorder=audit_recorder,
    )


def _normalized_scope(document_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({str(document_id) for document_id in document_ids if str(document_id)}))


def _document_version(row: dict[str, object]) -> int:
    value = row.get("document_version", 1)
    try:
        return int(cast("int | str", value))
    except (TypeError, ValueError):
        return 1


def _pipeline_version(row: dict[str, object]) -> str | None:
    explicit = row.get("pipeline_version")
    if explicit is not None:
        return str(explicit)
    for key in ("chunk_metadata", "document_metadata"):
        value = row.get(key)
        if isinstance(value, dict):
            version = value.get("pipeline_version") or value.get("extraction_version")
            if version is not None:
                return str(version)
    return None


def _contract_index(row: dict[str, object]) -> int | None:
    """Read the explicit multi-contract identity, preserving legacy NULL."""
    value = row.get("contract_index")
    if value is None:
        value = _record_metadata(row).get("contract_index")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _record_metadata(row: dict[str, object]) -> dict[str, object]:
    value = row.get("chunk_metadata")
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _bounded_snippet(content: str, maximum: int) -> str:
    if len(content) <= maximum:
        return content
    return content[: max(1, maximum - 1)].rstrip() + "…"


def _evidence_ref(
    *,
    handle: EvidenceHandle,
    graph_indexed: bool,
    score: float,
    rank: int,
    page_no: int,
    graph_indexed_at: datetime | None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=handle.evidence_id,
        chunk_id=handle.chunk_id,
        document_id=handle.document_id,
        document_version=handle.document_version,
        contract_index=handle.contract_index,
        pipeline_version=handle.pipeline_version,
        page_no=page_no,
        graph_indexed=graph_indexed,
        graph_indexed_at=graph_indexed_at,
        retrieval_method="hybrid_rrf" if rank else "handle_read",
        score=score,
        rank=rank,
        run_id=handle.run_id,
        thread_id=handle.thread_id,
        checkpoint_id=handle.checkpoint_id,
    )


def _graph_indexed_at(row: dict[str, object]) -> datetime | None:
    value = row.get("graph_indexed_at")
    return value if isinstance(value, datetime) else None


def _ingestion_state(record: dict[str, object], document_id: str) -> IngestionDocumentState:
    raw_status = str(record.get("status") or "received").lower()
    chunk_count = _coerce_int(record.get("chunk_count"))
    warning_count = _warning_count(record.get("warnings"))
    extraction_incomplete = bool(record.get("extraction_incomplete"))
    if raw_status in {"failed", "error", "rejected"}:
        state = "failed"
    elif raw_status in {"completed", "complete", "succeeded", "ready"} and chunk_count > 0:
        state = "completed_with_warnings" if warning_count or extraction_incomplete else "ready"
    else:
        state = "index_not_ready"
    return IngestionDocumentState(
        document_id=document_id,
        state=state,
        status=raw_status,
        chunk_count=chunk_count,
        warning_count=warning_count,
        graph_indexed_chunk_count=_coerce_int(record.get("verified_chunk_count")),
        document_version=(
            _coerce_int(record["document_version"])
            if record.get("document_version") is not None
            else None
        ),
        pipeline_version=_pipeline_version(record),
        graph_indexed_at=_graph_indexed_at(record),
    )


def _warning_count(raw: object) -> int:
    if not isinstance(raw, list):
        return 0
    return sum(len(item) if isinstance(item, list) else 1 for item in raw)


def _coerce_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _handle_matches_scope(
    *,
    handle: object,
    principal_id: str,
    tenant_id: str,
    thread_id: str,
    run_id: str,
    checkpoint_id: str | None,
    current_scope: tuple[str, ...],
) -> bool:
    if not isinstance(handle, EvidenceHandle):
        return False
    identity_matches = (
        handle.tenant_id == tenant_id
        and handle.principal_id == principal_id
        and handle.thread_id == thread_id
        and handle.run_id == run_id
    )
    checkpoint_matches = handle.checkpoint_id == checkpoint_id
    document_matches = (
        handle.allowed_doc_ids == current_scope and handle.document_id in current_scope
    )
    return identity_matches and checkpoint_matches and document_matches
