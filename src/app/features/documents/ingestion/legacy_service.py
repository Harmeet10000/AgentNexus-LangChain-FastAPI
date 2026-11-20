"""
IngestionService: runs IngestionGraph for a given uploaded document.

Called by the HTTP router. Non-streaming — awaits completion before returning.
If IngestionGraph returns an expected typed failure, maps it to the existing
project exception boundary before returning to FastAPI.

**State-key boundary.** ``DocumentIngestionState`` is a ``total=False``
``TypedDict`` and LangGraph maps an invocation's input onto declared channels,
silently discarding the rest. The legacy compatibility path historically sent
``doc_id``/``raw_bytes``/``thread_id``/``source``/``document_type``/
``jurisdiction`` — six keys the state does not declare. Every key is now placed
explicitly (``map_legacy_state``) or reported as unmapped in a typed failure; no
invocation reaches ``ainvoke`` with an undeclared key.
"""

from __future__ import annotations

import hashlib
import mimetypes
from typing import TYPE_CHECKING
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph
from returns.result import Failure, Success

from app.features.documents.ingestion.legacy_dto import DocumentUploadResponse
from app.features.documents.ingestion.legacy_errors import (
    IngestionGraphError,
    IngestionInternalError,
    IngestionPipelineError,
    IngestionValidationError,
)
from app.kernel.errors import FeatureError
from app.kernel.tenancy import personal_tenant_id
from app.observability.logging import logger, trace_layer
from app.utils.storage import build_s3_key

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from app.features.documents.ingestion.legacy_errors import IngestionResult
    from app.utils.storage import StorageResult, StorageService

#: Legacy key -> declared ``DocumentIngestionState`` channel. The mapping is
#: deliberately explicit: a new legacy key must be placed here or it fails loudly.
_DECLARED_STATE_PLACEMENTS: dict[str, str] = {
    "doc_id": "document_id",
    "user_id": "user_id",
    "tenant_id": "tenant_id",
    "filename": "filename",
    "content_type": "content_type",
    "object_uri": "object_uri",
    "document_version": "document_version",
    "attempt": "attempt",
    "document_type": "document_kind",
}

#: Legacy keys that are invocation metadata, not graph state. They travel in
#: ``RunnableConfig.configurable`` so they never reach the channel mapping.
_INVOCATION_CONFIG_PLACEMENTS: frozenset[str] = frozenset({"thread_id", "source", "jurisdiction"})

#: Bytes are not a state channel; they reach the ingest function by object-store
#: reference. They count as placed only once ``object_uri`` is present.
_RAW_BYTES_KEY = "raw_bytes"


def map_legacy_state(
    legacy: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    """Map a legacy state shape onto declared channels, config, and leftovers.

    Pure and total: returns ``(state, configurable, unmapped_keys)``. The caller
    must fail loudly when ``unmapped_keys`` is non-empty — silently dropping
    them is the defect this boundary contract exists to prevent.
    """
    state = {
        target: legacy[source]
        for source, target in _DECLARED_STATE_PLACEMENTS.items()
        if source in legacy
    }
    configurable = {key: legacy[key] for key in _INVOCATION_CONFIG_PLACEMENTS if key in legacy}
    unmapped = [
        key
        for key in legacy
        if key not in _DECLARED_STATE_PLACEMENTS and key not in _INVOCATION_CONFIG_PLACEMENTS
    ]
    if _RAW_BYTES_KEY in unmapped and "object_uri" in state:
        # The bytes were routed to object storage before mapping; the reference
        # is the declared channel and the raw payload is not needed in state.
        unmapped.remove(_RAW_BYTES_KEY)
    return state, configurable, sorted(unmapped)


class IngestionService:
    def __init__(
        self,
        ingestion_graph: CompiledStateGraph[Any],
        *,
        object_store: StorageService | None = None,
    ) -> None:
        self._graph = ingestion_graph
        self._object_store = object_store

    @trace_layer("service")
    async def ingest_document(
        self,
        raw_bytes: bytes,
        user_id: str,
        filename: str,
        source: str,
        *,
        content_type: str = "",
        document_type: str = "unknown",
        jurisdiction: str = "India",
        doc_id: str | None = None,
    ) -> IngestionResult[DocumentUploadResponse]:
        resolved_doc_id = doc_id or str(uuid4())
        thread_id = str(uuid4())  # ingestion gets its own thread_id
        tenant_id = personal_tenant_id(user_id)
        resolved_content_type = content_type or (
            mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )

        log = logger.bind(
            doc_id=resolved_doc_id,
            user_id=user_id,
            document_type=document_type,
        )
        log.info("ingestion_started")

        legacy_state: dict[str, object] = {
            "doc_id": resolved_doc_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "filename": filename,
            "content_type": resolved_content_type,
            "document_version": 1,
            "attempt": 0,
            "document_type": document_type,
            "thread_id": thread_id,
            "source": source,
            "jurisdiction": jurisdiction,
            "raw_bytes": raw_bytes,
        }
        if self._object_store is not None:
            object_uri = await self._store_raw_bytes(
                object_store=self._object_store,
                raw_bytes=raw_bytes,
                content_type=resolved_content_type,
                user_id=user_id,
                tenant_id=tenant_id,
                document_id=resolved_doc_id,
                filename=filename,
            )
            if isinstance(object_uri, Failure):
                error = object_uri.failure()
                log.error("ingestion_object_store_failed", error=error.message)
                return Failure(
                    IngestionGraphError(
                        message="Document ingestion failed to store the upload",
                        details={
                            "doc_id": resolved_doc_id,
                            "storage_code": str(error.code),
                        },
                        source="ingestion_service",
                        doc_id=resolved_doc_id,
                    )
                )
            legacy_state["object_uri"] = object_uri.unwrap()

        initial_state, configurable, unmapped = map_legacy_state(legacy_state)
        if unmapped:
            # Fail loudly, naming every key: the fix is mechanical, and a silent
            # drop would make the document look ingested when it never was.
            log.error("ingestion_state_keys_unmapped", unmapped_keys=unmapped)
            return Failure(
                IngestionValidationError(
                    message="Legacy ingestion state contains keys the graph does not declare",
                    details={"doc_id": resolved_doc_id, "unmapped_keys": unmapped},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            )

        try:
            result = await self._graph.ainvoke(
                initial_state,
                config={"configurable": configurable},
            )
        except Exception as exc:
            log.exception("ingestion_graph_failed", error=str(exc))
            return Failure(
                IngestionGraphError(
                    message="Document ingestion failed",
                    details={"doc_id": resolved_doc_id, "error": str(exc)},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            )

        failure = result.get("failure")
        if failure is not None:
            if isinstance(failure, FeatureError):
                error = IngestionPipelineError(
                    message=failure.message,
                    details=failure.details or {"doc_id": resolved_doc_id},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            elif isinstance(failure, dict):
                error = IngestionPipelineError(
                    message=str(failure.get("message", "Document ingestion failed")),
                    details={"doc_id": resolved_doc_id, "failure": failure},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            else:
                error = IngestionInternalError(
                    message=str(failure),
                    details={"doc_id": resolved_doc_id, "failure": str(failure)},
                    source="ingestion_service",
                    doc_id=resolved_doc_id,
                )
            return Failure(error)

        log.info(
            "ingestion_completed",
            entities=len(result.get("stored_entity_ids", [])),
            clauses=len(result.get("stored_clause_ids", [])),
        )

        return Success(
            DocumentUploadResponse(
                doc_id=resolved_doc_id,
                status="completed",
                entity_count=len(result.get("stored_entity_ids", [])),
                clause_count=len(result.get("stored_clause_ids", [])),
                relationship_count=len(result.get("stored_relationship_ids", [])),
                dropped_entity_count=result.get("dropped_entity_count", 0),
            )
        )

    @staticmethod
    async def _store_raw_bytes(
        *,
        object_store: StorageService,
        raw_bytes: bytes,
        content_type: str,
        user_id: str,
        tenant_id: str,
        document_id: str,
        filename: str,
    ) -> StorageResult[str]:
        """Route the upload to object storage; the graph reads it by reference."""
        object_key = build_s3_key(
            prefix="documents",
            user_id=user_id,
            tenant_id=tenant_id,
            document_id=document_id,
            content_hash=hashlib.sha256(raw_bytes).hexdigest(),
            filename=filename,
        )
        return await object_store.put_object(
            key=object_key,
            data=raw_bytes,
            content_type=content_type,
            metadata={
                "user_id": user_id,
                "tenant_id": tenant_id,
                "document_id": document_id,
                "source": "legacy_ingestion_compatibility",
            },
        )
