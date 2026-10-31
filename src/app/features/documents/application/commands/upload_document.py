"""Unified document feature services."""

from __future__ import annotations

import hashlib
import traceback
from asyncio import TimeoutError as AsyncTimeoutError
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from graphiti_core.errors import GraphitiError
from neo4j.exceptions import Neo4jError
from returns.result import Failure, Success
from sqlalchemy import func, select

from app.ai.embeddings import EmbeddingTaskType, embed_texts
from app.config import get_settings
from app.events import with_outbox, with_terminal_outbox
from app.extraction.langextract.langextract_to_graph import GraphIngestionContext
from app.extraction.langextract.relationships import (
    RELATIONSHIP_BUDGET_EXHAUSTED_REASON,
    RELATIONSHIP_CONFIG_DISABLED_REASON,
    ClauseIdentity,
    GroundedRelationship,
    RelationshipPassResult,
    prompt_safe_attribution,
    relationship_edge_id,
    relationship_pass_config_from_settings,
    run_relationship_pass,
)
from app.extraction.langextract.service import (
    ExtractionBudget,
    ExtractionFailed,
    ExtractionFailureCode,
    ExtractionRequest,
    ExtractionSucceeded,
)
from app.features.agent_saul.domain.state import (
    Citation,
    ClauseSegment,
    ClauseType,
    LegalRelationship,
)
from app.features.auth.domain.tenancy import personal_tenant_id
from app.features.documents.adapters.persistence.postgres.document_repository import (
    DocumentRepository,
    build_chunk_rows,
)
from app.features.documents.api.http.dto import (
    DocumentStatusResponse,
    DocumentUploadResponse,
    QualityWarningDTO,
)
from app.features.documents.application.access_audit import (
    DocumentAccessAuditEvent,
    DocumentAccessAuditRecorder,
    InMemoryDocumentAccessAuditRecorder,
)
from app.features.documents.application.queries.documents import (
    _KNOWLEDGE_EXTRACTION_PROMPT,
    _batched,
    _flatten_warnings,
    _LegalPreparation,
)
from app.features.documents.domain.constants import (
    ANALYZE_THRESHOLD_CHUNKS,
    INGEST_EMBEDDING_BATCH_SIZE,
)
from app.features.documents.domain.errors import (
    DocumentGraphWriteError,
    DocumentIngestionCheckpointError,
    DocumentNotFoundError,
    DocumentStorageError,
    DocumentValidationError,
)
from app.features.documents.domain.fidelity import (
    GraphRelationshipsState,
    GraphRelationshipsStatus,
    GraphWritePhaseResult,
    IngestionFidelityMetrics,
    aggregate_relationship_states,
)
from app.features.documents.ingestion.adapters.graphiti.verifier import write_and_verify_chunk
from app.features.documents.ingestion.chunking.classification import (
    ClassifiedDocument,
    ContractBoundary,
    ParsedDocument,
    QualityWarning,
    classify_document,
    establish_contract_boundaries,
    segment_chunks,
)
from app.features.documents.ingestion.chunking.fidelity import (
    BoundaryFidelityInputs,
    build_fidelity_metrics,
    fidelity_degradation_warnings,
)
from app.features.documents.ingestion.enrichment.legal_metadata import (
    contract_event_dates,
    enrich_legal_chunks,
    extract_legal_metadata,
)
from app.features.documents.ingestion.parsing.parser import parse_document
from app.kernel.result import log_expected_failure
from app.observability.logging import logger, trace_layer
from app.retrieval.graphiti.grouping import build_graphiti_group_id
from app.retrieval.graphiti.write_clause_episodes import write_clause_episodes_to_graphiti
from app.shared.jobs.names import DOCUMENTS_INGEST
from app.utils.storage import build_s3_key, key_from_s3_uri

_GRAPHITI_TRANSPORT_ERRORS = (GraphitiError, Neo4jError, AsyncTimeoutError, OSError)
_GRAPHITI_EVENT_ERRORS = (
    *_GRAPHITI_TRANSPORT_ERRORS,
    AttributeError,
    TypeError,
    ValueError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from graphiti_core.graphiti import Graphiti
    from langgraph.graph.state import CompiledStateGraph
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, async_sessionmaker

    from app.features.documents.api.http.dto import IngestionJob, IngestionRuntime
    from app.features.documents.domain.errors import DocumentResult
    from app.features.documents.ingestion.chunking.classification import PreparedChunk
    from app.features.documents.ingestion.enrichment.legal_metadata import (
        LegalMetadataExtraction,
    )
    from app.resilience.idempotency import IdempotencyGuard
    from app.retrieval.graphiti.write_clause_episodes import (
        GraphitiService as ClauseWriterService,
    )
    from app.utils.storage import StorageService


class DocumentCommandService:
    """Create, queue, and inspect document ingestion jobs."""

    def __init__(
        self,
        repo: DocumentRepository,
        object_store: StorageService | None,
        audit_recorder: DocumentAccessAuditRecorder | None = None,
    ):
        self.repo: DocumentRepository = repo
        self.object_store: StorageService | None = object_store
        self.audit_recorder = audit_recorder or InMemoryDocumentAccessAuditRecorder()

    @trace_layer("service")
    async def upload_document(
        self,
        *,
        user_id: str,
        tenant_id: str | None = None,
        filename: str,
        content_type: str,
        raw_bytes: bytes,
    ) -> DocumentResult[DocumentUploadResponse]:
        if not raw_bytes:
            return Failure(
                DocumentValidationError(
                    message="Uploaded document is empty", source="document_service"
                )
            )
        tenant_id = tenant_id or personal_tenant_id(user_id)
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        existing_result = await self.repo.get_document_by_user_hash(
            user_id=user_id, tenant_id=tenant_id, content_hash=content_hash
        )
        if isinstance(existing_result, Success):
            existing = existing_result.unwrap()
            if existing is not None:
                return Success(
                    DocumentUploadResponse(
                        doc_id=str(existing.id),
                        status=existing.status,
                        duplicate=True,
                    )
                )
        elif isinstance(existing_result, Failure):
            failure = existing_result.failure()
            # A miss on the duplicate check is the NORMAL first-upload path,
            # not an error — only real infrastructure failures abort here.
            if not isinstance(failure, DocumentNotFoundError):
                log_expected_failure(error=failure, operation="document_upload")
                return Failure(failure)

        document_id = str(object=uuid4())
        object_key = build_s3_key(
            prefix="documents",
            user_id=user_id,
            tenant_id=tenant_id,
            document_id=document_id,
            content_hash=content_hash,
            filename=filename,
        )
        if self.object_store is None:
            return Failure(
                DocumentStorageError(
                    message="Object storage is not configured", source="document_service"
                )
            )
        storage_result = await self.object_store.put_object(
            key=object_key,
            data=raw_bytes,
            content_type=content_type,
            metadata={
                "user_id": user_id,
                "tenant_id": tenant_id,
                "document_id": document_id,
                "content_hash": content_hash,
            },
        )
        if isinstance(storage_result, Failure):
            error = storage_result.failure()
            return Failure(
                DocumentStorageError(
                    message=error.message,
                    details=error.details,
                    source="object_storage",
                )
            )
        object_uri = storage_result.unwrap()
        create_result = await self.repo.create_document(
            user_id=user_id,
            title=filename,
            source_uri=filename,
            object_uri=object_uri,
            content_hash=content_hash,
            document_kind="generic",
            status="received",
            jurisdiction=None,
            contract_type=None,
            parties=[],
            metadata_={"content_type": content_type, "filename": filename},
        )
        if isinstance(create_result, Failure):
            log_expected_failure(error=create_result.failure(), operation="document_upload")
            return Failure(create_result.failure())
        document = create_result.unwrap()

        await with_outbox(
            session=self.repo.session,
            aggregate_type="user_document",
            aggregate_id=str(object=document.id),
            event_type=DOCUMENTS_INGEST,
            payload={
                "document_id": str(object=document.id),
                "user_id": user_id,
                "tenant_id": tenant_id,
                "filename": filename,
                "content_type": content_type,
                "object_uri": object_uri,
                "document_version": 1,
            },
        )

        logger.bind(document_id=str(object=document.id)).info("documents_ingest_queued")
        return Success(
            DocumentUploadResponse(
                doc_id=str(object=document.id),
                status="queued",
                task_id=None,
                object_uri=object_uri,
                document_kind=document.document_kind,
                warning_count=0,
            )
        )

    @trace_layer("service")
    async def get_status(
        self, *, user_id: str, document_id: str, tenant_id: str | None = None
    ) -> DocumentResult[DocumentStatusResponse]:
        resolved_tenant = tenant_id or personal_tenant_id(user_id)
        result = await self._get_status(
            user_id=user_id, document_id=document_id, tenant_id=resolved_tenant
        )
        outcome = "ok" if isinstance(result, Success) else "denied"
        try:
            await self.audit_recorder.append(
                DocumentAccessAuditEvent(
                    tenant_id=resolved_tenant,
                    principal_id=user_id,
                    operation="ingestion.status",
                    requested_document_ids=(document_id,),
                    document_ids=(document_id,) if isinstance(result, Success) else (),
                    policy="mechanically_verified",
                    outcome=outcome,
                    reason=None if isinstance(result, Success) else result.failure().message,
                )
            )
        except Exception as exc:  # noqa: BLE001 — audit is a fail-closed boundary
            exc.add_note(f"operation=ingestion.status, document_id={document_id}")
            logger.bind(document_id=document_id, user_id=user_id).exception(
                "document_access_audit_failed"
            )
        return result

    async def _get_status(
        self, *, user_id: str, document_id: str, tenant_id: str
    ) -> DocumentResult[DocumentStatusResponse]:
        status_result = await self.repo.fetch_status(
            user_id=user_id,
            tenant_id=tenant_id,
            document_id=document_id,
        )
        if isinstance(status_result, Success):
            record: dict[str, Any] | None = status_result.unwrap()
            if record is not None:
                warnings: list[QualityWarningDTO] = _flatten_warnings(record.get("warnings", []))
                raw_metadata = record.get("document_metadata")
                metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
                return Success(
                    DocumentStatusResponse(
                        doc_id=str(object=record["document_id"]),
                        status=str(object=record["status"]),
                        object_uri=str(object=record["object_uri"]),
                        title=str(object=record["title"]),
                        document_kind=str(object=record["document_kind"]),
                        chunk_count=int(record["chunk_count"]),
                        verified_chunk_count=int(record["verified_chunk_count"]),
                        warning_count=len(warnings),
                        warnings=warnings,
                        graph_relationships=_relationship_state_from_metadata(metadata),
                        fidelity=_fidelity_metrics_from_metadata(metadata),
                    )
                )
        if isinstance(status_result, Failure):
            return Failure(status_result.failure())
        return Failure(
            DocumentNotFoundError(
                message="Document not found",
                details={"document_id": document_id},
                source="document_service",
            )
        )


async def _load_document_bytes(
    object_store: StorageService, object_uri: str
) -> DocumentResult[bytes]:
    key_result = key_from_s3_uri(object_uri)
    if isinstance(key_result, Failure):
        error = key_result.failure()
        return Failure(
            DocumentValidationError(
                message=error.message,
                details=error.details,
                source="object_storage",
            )
        )
    object_result = await object_store.get_object(key=key_result.unwrap())
    if isinstance(object_result, Failure):
        return Failure(
            DocumentStorageError(
                message=object_result.failure().message,
                details=object_result.failure().details,
                source="object_storage",
            )
        )
    return Success(object_result.unwrap())


@trace_layer("service")
async def process_document_ingestion(  # noqa: PLR0912, PLR0914, PLR0915 — staged workflow
    *,
    job: IngestionJob,
    runtime: IngestionRuntime,
) -> DocumentResult[dict[str, object]]:
    raw_result = await _load_document_bytes(runtime.object_store, job.object_uri)
    if isinstance(raw_result, Failure):
        return raw_result
    raw_bytes = raw_result.unwrap()
    settings = get_settings()
    parsed: ParsedDocument = await parse_document(
        raw_bytes=raw_bytes,
        filename=job.filename,
        content_type=job.content_type,
        extraction_timeout_seconds=settings.DOCUMENT_EXTRACTION_TIMEOUT_SECONDS,
        max_pages=settings.DOCUMENT_EXTRACTION_MAX_PAGES,
        max_output_bytes=settings.DOCUMENT_EXTRACTION_MAX_OUTPUT_BYTES,
    )
    boundary_result = establish_contract_boundaries(
        markdown=parsed.markdown, page_count=parsed.page_count
    )
    boundaries = boundary_result.boundaries or [
        ContractBoundary(
            contract_index=0,
            markdown=parsed.markdown,
            start_char=0,
            end_char=len(parsed.markdown),
        )
    ]
    # Classification and legal extraction are intentionally inside this loop:
    # a whole-file metadata pass is unsound for a PDF containing two contracts.
    contract_classifications: list[ClassifiedDocument] = []
    legal_preparations: list[tuple[ContractBoundary, _LegalPreparation]] = []
    contract_parsed_items: list[tuple[ContractBoundary, ParsedDocument, ClassifiedDocument]] = []
    prepared_chunks: list[PreparedChunk] = []
    segmentation_warnings: list[QualityWarning] = []
    for boundary in boundaries:
        contract_parsed = parsed.model_copy(update={"markdown": boundary.markdown})
        contract_classified = classify_document(
            markdown=contract_parsed.markdown, filename=job.filename
        )
        contract_classifications.append(contract_classified)
        contract_parsed_items.append((boundary, contract_parsed, contract_classified))
        legal_preparation = (
            await _prepare_legal_metadata(
                runtime=runtime,
                markdown=contract_parsed.markdown,
                classified=contract_classified,
            )
            if not boundary_result.ambiguous
            else _LegalPreparation(metadata=None, warnings=[])
        )
        legal_preparations.append((boundary, legal_preparation))

    classified = _aggregate_contract_classifications(
        contract_classifications,
        boundary_warnings=boundary_result.warnings,
        ambiguous=boundary_result.ambiguous,
    )
    legal = (
        legal_preparations[0][1]
        if len(legal_preparations) == 1
        else _LegalPreparation(
            metadata=None,
            warnings=[
                warning for _, preparation in legal_preparations for warning in preparation.warnings
            ],
        )
    )
    extraction_outcome, relationship_state = await _extract_document_knowledge(
        runtime=runtime,
        job=job,
        boundaries=boundaries,
        boundary_ambiguous=boundary_result.ambiguous,
        jurisdiction=(
            legal.metadata.jurisdiction
            if legal.metadata
            else (classified.jurisdiction if len(boundaries) == 1 else None)
        ),
        document_type=classified.document_kind,
    )
    status_result = await runtime.repo.update_document_status(
        document_id=job.document_id,
        status="failed" if _is_graph_write_failure(extraction_outcome) else "parsed",
        title=parsed.title,
        document_kind=classified.document_kind,
        jurisdiction=(legal.metadata.jurisdiction if legal.metadata else classified.jurisdiction),
        contract_type=(
            legal.metadata.contract_type
            if legal.metadata
            else (classified.contract_type if len(boundaries) == 1 else None)
        ),
        parties=[
            *(
                legal.metadata.parties
                if legal.metadata
                else (classified.parties if len(boundaries) == 1 else [])
            )
        ],
        metadata_={
            "content_type": job.content_type,
            "filename": job.filename,
            **classified.metadata_,
            **(legal.metadata.model_dump(exclude_none=True) if legal.metadata else {}),
            **(
                {
                    "contracts": [
                        {
                            "contract_index": boundary.contract_index,
                            **(
                                preparation.metadata.model_dump(exclude_none=True)
                                if preparation.metadata
                                else {}
                            ),
                        }
                        for boundary, preparation in legal_preparations
                    ]
                }
                if len(boundaries) > 1
                else {}
            ),
            **_extraction_metadata(
                extraction_outcome,
                document_version=job.document_version,
            ),
            # The relationship-phase truth is persisted as soon as it is known,
            # so a crash during chunking still leaves the record truthful.
            "graph_relationships": relationship_state.model_dump_for_metadata(),
        },
        structural_tree=parsed.structural_tree,
        extraction_incomplete=isinstance(extraction_outcome, ExtractionFailed),
    )
    if isinstance(status_result, Failure):
        return Failure(status_result.failure())
    if isinstance(extraction_outcome, ExtractionFailed) and _is_graph_write_failure(
        extraction_outcome
    ):
        return Failure(
            DocumentGraphWriteError(
                message=extraction_outcome.message,
                details={
                    "document_id": job.document_id,
                    "extraction_budget": extraction_outcome.budget.model_dump(mode="json"),
                    "knowledge_extraction_failure": str(extraction_outcome.code),
                    "graph_relationships": relationship_state.model_dump_for_metadata(),
                },
                source="graphiti",
            )
        )
    checkpoint_result = await _checkpoint_ingestion_transaction(runtime)
    if isinstance(checkpoint_result, Failure):
        return Failure(checkpoint_result.failure())
    # Chunking happens only after extraction/status checkpoint.  This preserves
    # the old transactional ordering while keeping each contract isolated.
    boundary_chunks: list[tuple[ContractBoundary, list[PreparedChunk]]] = []
    for index, (boundary, contract_parsed, contract_classified) in enumerate(contract_parsed_items):
        contract_chunks, warnings = await segment_chunks(
            parsed=contract_parsed,
            classified=contract_classified,
        )
        legal_preparation = legal_preparations[index][1]
        if legal_preparation.metadata is not None:
            contract_chunks = enrich_legal_chunks(
                chunks=contract_chunks,
                classified=contract_classified,
                metadata=legal_preparation.metadata,
            )
        annotated = _annotate_contract_chunks(contract_chunks, boundary)
        prepared_chunks.extend(annotated)
        boundary_chunks.append((boundary, annotated))
        segmentation_warnings.extend(warnings)

    # Chunk identity remains unique within the document while contract_index
    # preserves the independent legal identity used by evidence citations.
    prepared_chunks = [
        chunk.model_copy(
            update={
                "chunk_index": index,
                "document_version": job.document_version,
            }
        )
        for index, chunk in enumerate(prepared_chunks)
    ]
    # Fidelity is measured where the information exists: after chunk annotation
    # (source spans) and before embedding (so degradation warnings land on the
    # persisted chunk warnings).  Threshold breaches degrade the terminal state.
    fidelity_metrics = build_fidelity_metrics(
        parsed=parsed,
        boundaries=[
            BoundaryFidelityInputs(
                markdown=boundary_entry.markdown,
                chunks=tuple(chunks),
                document_kind=contract_parsed_items[index][2].document_kind,
            )
            for index, (boundary_entry, chunks) in enumerate(boundary_chunks)
        ],
        chunk_count=len(prepared_chunks),
        document_kind=classified.document_kind,
    )
    fidelity_warnings = fidelity_degradation_warnings(fidelity_metrics)
    chunk_rows = await _embed_chunks(
        user_id=job.user_id,
        document_id=job.document_id,
        chunks=prepared_chunks,
        extra_warnings=(
            boundary_result.warnings
            + segmentation_warnings
            + classified.warnings
            + legal.warnings
            + fidelity_warnings
        ),
        extraction_outcome=extraction_outcome,
        document_version=job.document_version,
    )
    upsert_result = await runtime.repo.upsert_chunks(
        build_chunk_rows(
            document_id=job.document_id,
            user_id=job.user_id,
            tenant_id=job.tenant_id,
            chunks=chunk_rows,
        )
    )
    if isinstance(upsert_result, Failure):
        log_expected_failure(upsert_result.failure(), operation="document_ingestion")
        return Failure(upsert_result.failure())
    if len(chunk_rows) > ANALYZE_THRESHOLD_CHUNKS:
        analyze_result = await runtime.repo.analyze_chunks()
        if isinstance(analyze_result, Failure):
            return Failure(analyze_result.failure())
    status_result = await runtime.repo.update_document_status(
        document_id=job.document_id, status="stored_postgres"
    )
    if isinstance(status_result, Failure):
        return Failure(status_result.failure())
    checkpoint_result = await _checkpoint_ingestion_transaction(runtime)
    if isinstance(checkpoint_result, Failure):
        return Failure(checkpoint_result.failure())
    if classified.graphiti_required:
        for boundary, preparation in legal_preparations:
            await _write_contract_events(
                graphiti=runtime.graphiti,
                legal_metadata=preparation.metadata,
                document_id=job.document_id,
                contract_index=boundary.contract_index,
                tenant_id=job.tenant_id,
                document_version=job.document_version,
            )
        verify_result = await _verify_legal_chunks(
            repo=runtime.repo,
            graphiti=runtime.graphiti,
            user_id=job.user_id,
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            chunk_rows=chunk_rows,
        )
        if isinstance(verify_result, Failure):
            return Failure(verify_result.failure())
        verified_count = verify_result.unwrap()
        final_status: Literal["completed", "completed_with_warnings"] = (
            "completed"
            if verified_count == len(chunk_rows)
            and not isinstance(extraction_outcome, ExtractionFailed)
            and not fidelity_warnings
            else "completed_with_warnings"
        )
    else:
        final_status = (
            "completed"
            if not segmentation_warnings
            and not classified.warnings
            and not isinstance(extraction_outcome, ExtractionFailed)
            and not fidelity_warnings
            else "completed_with_warnings"
        )
        verified_count = 0
    status_result = await runtime.repo.update_document_status(
        document_id=job.document_id,
        status=final_status,
        metadata_={
            "ingestion_state": {
                "tenant_id": job.tenant_id,
                "document_id": job.document_id,
                "document_version": job.document_version,
                "attempt": job.attempt,
                "task_id": job.task_id or "",
            },
            # Re-assert the relationship truth (the earlier write already
            # carried it; the shallow jsonb merge replaces the same key) and
            # persist the measured fidelity metrics with their thresholds.
            "graph_relationships": relationship_state.model_dump_for_metadata(),
            "fidelity": fidelity_metrics.model_dump(mode="json"),
        },
    )
    if isinstance(status_result, Failure):
        return Failure(status_result.failure())
    session = getattr(runtime.repo, "session", None)
    if session is not None:
        await with_terminal_outbox(
            session,
            document_id=job.document_id,
            payload={
                "document_id": job.document_id,
                "user_id": job.user_id,
                "tenant_id": job.tenant_id,
                "status": final_status,
                "document_version": job.document_version,
                "attempt": job.attempt,
                "task_id": job.task_id,
                "graph_relationships_status": relationship_state.status.value,
            },
        )
    return Success(
        {
            "status": final_status,
            "document_id": job.document_id,
            "chunk_count": len(chunk_rows),
            "verified_chunk_count": verified_count,
            "document_kind": classified.document_kind,
            "graph_relationships_status": str(object=relationship_state.status.value),
            "fidelity_warning_count": len(fidelity_warnings),
        }
    )


def _relationship_state_from_metadata(
    metadata: dict[str, Any],
) -> GraphRelationshipsState | None:
    """Decode the persisted relationship-phase truth, or None when absent/invalid."""
    raw = metadata.get("graph_relationships")
    if not isinstance(raw, dict):
        return None
    try:
        return GraphRelationshipsState.model_validate(raw)
    except ValueError:
        logger.bind(operation="decode_graph_relationships_metadata").warning(
            "document_metadata_graph_relationships_invalid"
        )
        return None


def _fidelity_metrics_from_metadata(
    metadata: dict[str, Any],
) -> IngestionFidelityMetrics | None:
    """Decode persisted fidelity metrics, or None when absent/invalid."""
    raw = metadata.get("fidelity")
    if not isinstance(raw, dict):
        return None
    try:
        return IngestionFidelityMetrics.model_validate(raw)
    except ValueError:
        logger.bind(operation="decode_fidelity_metadata").warning(
            "document_metadata_fidelity_invalid"
        )
        return None


async def _checkpoint_ingestion_transaction(runtime: IngestionRuntime) -> DocumentResult[None]:
    """Commit database work before entering another potentially slow external phase."""
    checkpoint = getattr(runtime, "transaction_checkpoint", None)
    if checkpoint is None:
        return Success(None)
    try:
        await checkpoint()
    except Exception as exc:  # noqa: BLE001 — injected checkpoint exposes driver errors
        exc.add_note("operation=document_ingestion_checkpoint")
        return Failure(
            DocumentIngestionCheckpointError(
                message="Document ingestion database checkpoint failed",
                details={"error": str(exc)},
                source="postgres",
            )
        )
    return Success(None)


def _annotate_contract_chunks(
    chunks: list[PreparedChunk], boundary: ContractBoundary
) -> list[PreparedChunk]:
    """Attach stable contract and source coordinates without inventing spans."""
    annotated: list[PreparedChunk] = []
    cursor = 0
    for chunk in chunks:
        start = boundary.markdown.find(chunk.content, cursor)
        if start < 0:
            start = boundary.markdown.find(chunk.content)
        end = start + len(chunk.content) if start >= 0 else None
        absolute_start = boundary.start_char + start if start >= 0 else None
        absolute_end = boundary.start_char + end if end is not None else None
        if end is not None:
            cursor = end
        metadata = {
            **chunk.metadata_,
            "contract_index": boundary.contract_index,
            "source_start_char": absolute_start,
            "source_end_char": absolute_end,
            "source_boundary_start_char": boundary.start_char,
            "source_boundary_end_char": boundary.end_char,
        }
        annotated.append(
            chunk.model_copy(
                update={
                    "contract_index": boundary.contract_index,
                    "source_start_char": absolute_start,
                    "source_end_char": absolute_end,
                    "metadata_": metadata,
                }
            )
        )
    return annotated


def _aggregate_contract_classifications(
    classifications: list[ClassifiedDocument],
    *,
    boundary_warnings: list[QualityWarning],
    ambiguous: bool,
) -> ClassifiedDocument:
    if not classifications:
        return ClassifiedDocument(document_kind="generic", warnings=list(boundary_warnings))
    first = classifications[0]
    is_multi = len(classifications) > 1
    warnings = [
        *boundary_warnings,
        *(warning for classification in classifications for warning in classification.warnings),
    ]
    if ambiguous:
        warnings.append(
            QualityWarning(
                stage="contract_boundaries",
                code="MULTI_CONTRACT_BOUNDARY_AMBIGUOUS",
                message="Legal metadata extraction was skipped because contract boundaries were ambiguous.",
                severity="warning",
            )
        )
    if not is_multi:
        return first.model_copy(update={"warnings": warnings})
    kinds = {classification.document_kind for classification in classifications}
    return ClassifiedDocument(
        document_kind="legal_contract" if "legal_contract" in kinds else first.document_kind,
        graphiti_required=any(
            classification.graphiti_required for classification in classifications
        ),
        warnings=warnings,
        metadata_={
            "contract_count": len(classifications),
            "contract_kinds": sorted(kinds),
        },
    )


async def _prepare_legal_metadata(
    *,
    runtime: IngestionRuntime,
    markdown: str,
    classified: ClassifiedDocument,
) -> _LegalPreparation:
    if not classified.graphiti_required:
        return _LegalPreparation(metadata=None, warnings=[])
    metadata, warnings = await extract_legal_metadata(
        llm=runtime.llm,
        markdown=markdown,
        classified=classified,
    )
    return _LegalPreparation(metadata=metadata, warnings=warnings)


async def _write_contract_events(
    graphiti: Graphiti | None,
    legal_metadata: LegalMetadataExtraction | None,
    *,
    document_id: str,
    contract_index: int | None = None,
    tenant_id: str = "",
    document_version: int = 1,
) -> None:
    if graphiti is None or legal_metadata is None:
        return
    for event_name, event_date in contract_event_dates(legal_metadata):
        try:
            await graphiti.add_episode(
                name=f"{event_name}:{document_id}:{contract_index}:{event_date}",
                episode_body=(
                    f"{event_name} for {document_id} contract {contract_index} "
                    f"occurs on {event_date}."
                ),
                source_description=(
                    "{"
                    f'"doc_id":"{document_id}",'
                    f'"contract_index":{contract_index if contract_index is not None else "null"},'
                    f'"event_type":"{event_name}",'
                    f'"event_date":"{event_date}"'
                    "}"
                ),
                reference_time=None,  # ty: ignore[invalid-argument-type]
                group_id=build_graphiti_group_id(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    document_version=document_version,
                    contract_index=contract_index,
                ),
            )
        except _GRAPHITI_EVENT_ERRORS as exc:
            exc.add_note(
                f"document_id={document_id}, event_name={event_name}, operation=contract_event"
            )
            exc.add_note(traceback.format_exc())
            logger.bind(
                document_id=document_id,
                event_name=event_name,
                event_date=event_date,
                error_type=type(exc).__name__,
            ).exception("graphiti_event_episode_failed")


@trace_layer("service")
async def run_document_ingestion_task(
    *,
    document_id: str,
    user_id: str,
    tenant_id: str | None = None,
    filename: str,
    content_type: str,
    object_uri: str,
    task_id: str | None = None,
    attempt: int = 0,
    document_version: int = 1,
    graph: CompiledStateGraph[Any],
    engine: AsyncEngine,
    session_local: async_sessionmaker[Any],
    idempotency: IdempotencyGuard,
) -> dict[str, object]:
    tenant_id = tenant_id or personal_tenant_id(user_id)
    lock_key = _document_ingestion_lock_key(tenant_id, document_id, document_version)
    async with engine.connect() as lock_connection:
        acquired = await lock_connection.scalar(
            select(func.pg_try_advisory_lock(lock_key)),
        )
        await lock_connection.commit()
        if not acquired:
            return {"status": "skipped", "document_id": document_id}
        primary_error: BaseException | None = None
        try:
            # Reuse the checked-out connection that owns the session-level lock.
            # Opening a second connection here deadlocks against a supported
            # pool_size=1/max_overflow=0 deployment before ingestion can start.
            async with session_local(bind=lock_connection) as session:
                repo = DocumentRepository(session)
                job_idempotency = idempotency.for_session(session)
                attempt_checkpoint = await repo.update_document_status(
                    document_id=document_id,
                    status="processing",
                    metadata_={
                        "ingestion_state": {
                            "tenant_id": tenant_id,
                            "document_id": document_id,
                            "document_version": document_version,
                            "attempt": attempt,
                            "task_id": task_id or "",
                        }
                    },
                    extraction_incomplete=False,
                )
                if isinstance(attempt_checkpoint, Failure):
                    return {"status": "failed", "error_retryable": True}
                await session.commit()
                result = await graph.ainvoke(
                    {
                        "document_id": document_id,
                        "user_id": user_id,
                        "tenant_id": tenant_id,
                        "filename": filename,
                        "content_type": content_type,
                        "object_uri": object_uri,
                        "document_version": document_version,
                        "attempt": attempt,
                        "task_id": task_id,
                    },
                    {
                        "configurable": {
                            "document_repository": repo,
                            "document_idempotency": job_idempotency,
                            "task_id": task_id,
                            "attempt": attempt,
                        }
                    },
                )
                await session.commit()
                return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            await _release_document_ingestion_lock(
                connection=lock_connection,
                lock_key=lock_key,
                primary_error=primary_error,
            )


async def _release_document_ingestion_lock(
    *,
    connection: AsyncConnection,
    lock_key: int,
    primary_error: BaseException | None,
) -> None:
    """Release a session lock or discard its connection without masking failures."""
    try:
        released = await connection.scalar(
            select(func.pg_advisory_unlock(lock_key)),
        )
        await connection.commit()
        if not released:
            message = "Document ingestion advisory lock release failed"
            raise RuntimeError(message)  # noqa: TRY301 — handled by the invalidation path below
    except BaseException as cleanup_error:
        try:
            await connection.invalidate()
        except BaseException as invalidation_error:  # noqa: BLE001 — cleanup must not mask primary
            cleanup_error.add_note(
                f"Advisory-lock connection invalidation also failed: {invalidation_error!r}"
            )
        if primary_error is None:
            raise
        primary_error.add_note(f"Advisory-lock cleanup failed: {cleanup_error!r}")
        logger.bind(
            error_type=type(cleanup_error).__name__,
            lock_key=lock_key,
        ).exception("Document ingestion advisory-lock cleanup failed")


def _document_ingestion_lock_key(
    tenant_id: str,
    document_id: str,
    document_version: int,
) -> int:
    """Build a stable tenant/document/version-scoped advisory-lock key."""
    identity = f"{tenant_id}\0{document_id}\0{document_version}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], byteorder="big", signed=True)


async def _embed_chunks(
    *,
    user_id: str,
    document_id: str,
    chunks: list[PreparedChunk],
    extra_warnings: list[QualityWarning],
    extraction_outcome: ExtractionSucceeded | ExtractionFailed,
    document_version: int = 1,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch in _batched(chunks, INGEST_EMBEDDING_BATCH_SIZE):
        embeddings = await embed_texts(
            [f"{chunk.preamble}\n\n{chunk.content}".strip() for chunk in batch],
            task_type=EmbeddingTaskType.DOCUMENT,
        )
        for chunk, embedding in zip(batch, embeddings, strict=True):
            extraction_metadata = _extraction_metadata(
                extraction_outcome,
                document_version=document_version,
            )
            rows.append(
                {
                    "id": str(uuid4()),
                    "chunk_index": chunk.chunk_index,
                    "document_version": chunk.document_version,
                    "contract_index": chunk.contract_index,
                    "chunk_kind": chunk.chunk_kind,
                    "content": chunk.content,
                    "preamble": chunk.preamble,
                    "locus": chunk.locus,
                    "clause_type": chunk.clause_type,
                    "page_no": chunk.page_no,
                    "embedding": embedding,
                    "metadata_": chunk.metadata_,
                    "custom_metadata": {**chunk.custom_metadata, **extraction_metadata},
                    "quality_warnings": [
                        warning.model_dump()
                        for warning in [*chunk.quality_warnings, *extra_warnings]
                    ],
                }
            )
    _ = (user_id, document_id)
    return rows


def _extraction_metadata(
    outcome: ExtractionSucceeded | ExtractionFailed,
    *,
    document_version: int | None = None,
) -> dict[str, object]:
    if isinstance(outcome, ExtractionFailed):
        metadata: dict[str, object] = {
            "knowledge_extraction": "incomplete",
            "knowledge_extraction_failure": str(outcome.code),
            "knowledge_extraction_message": outcome.message,
        }
        if outcome.budget_dimension is not None:
            metadata["extraction_budget_dimension"] = str(outcome.budget_dimension)
        if outcome.budget is not None:
            metadata["extraction_budget"] = outcome.budget.model_dump(mode="json")
        if document_version is not None:
            metadata["knowledge_extraction_document_version"] = document_version
        return metadata
    classes = list(outcome.extraction_classes) or sorted(
        {
            extraction.extraction_class
            for document in outcome.documents
            for extraction in document.extractions
        }
    )
    metadata = {
        "knowledge_extraction": "complete",
        "knowledge_extraction_classes": classes,
    }
    metadata["extraction_budget"] = outcome.budget.model_dump(mode="json")
    if document_version is not None:
        metadata["knowledge_extraction_document_version"] = document_version
    return metadata


def _is_graph_write_failure(outcome: ExtractionSucceeded | ExtractionFailed) -> bool:
    return (
        isinstance(outcome, ExtractionFailed)
        and outcome.code is ExtractionFailureCode.GRAPH_WRITE_ERROR
    )


async def _extract_document_knowledge(  # noqa: PLR0912, PLR0914 — one extraction pass keeps all phase locals together
    *,
    runtime: IngestionRuntime,
    job: IngestionJob,
    boundaries: list[ContractBoundary],
    boundary_ambiguous: bool = False,
    jurisdiction: str | None,
    document_type: str,
) -> tuple[ExtractionSucceeded | ExtractionFailed, GraphRelationshipsState]:
    """Run pass-1 knowledge extraction plus the graph-write phase.

    Returns the extraction outcome together with the typed aggregate
    relationship-phase state, so callers can persist "not attempted" /
    "skipped (not implemented)" / "succeeded" / "failed" as distinct truths
    instead of an ``all([])`` success.
    """
    if boundary_ambiguous:
        return (
            ExtractionFailed(
                code=ExtractionFailureCode.BOUNDARY_AMBIGUOUS,
                message="Contract boundaries are ambiguous; legal extraction was skipped",
            ),
            GraphRelationshipsState(
                status=GraphRelationshipsStatus.NOT_ATTEMPTED,
                reason="boundary_ambiguous_extraction_skipped",
            ),
        )
    settings = get_settings()
    relationship_config = relationship_pass_config_from_settings(settings)
    budget = ExtractionBudget(
        max_contracts=settings.LANGEXTRACT_MAX_CONTRACTS,
        max_model_calls=settings.LANGEXTRACT_MAX_MODEL_CALLS,
        max_input_tokens=settings.LANGEXTRACT_MAX_INPUT_TOKENS,
        max_output_tokens=settings.LANGEXTRACT_MAX_OUTPUT_TOKENS,
        max_cost_usd=settings.LANGEXTRACT_MAX_COST_USD,
        cost_per_1k_tokens_usd=settings.LANGEXTRACT_COST_PER_1K_TOKENS_USD,
        wall_clock_seconds=settings.LANGEXTRACT_WALL_CLOCK_SECONDS,
        relationship_max_model_calls=relationship_config.max_model_calls,
        relationship_max_input_tokens=relationship_config.max_input_tokens,
        relationship_max_output_tokens=relationship_config.max_output_tokens,
        relationship_max_cost_usd=relationship_config.max_cost_usd,
    )
    # Rehydrate the checkpointed ledger before a Celery retry.  A retry must
    # not receive a fresh model-call/token budget merely because the worker
    # process was replaced.
    fetch_status = getattr(runtime.repo, "fetch_status", None)
    if fetch_status is not None:
        status_snapshot = await fetch_status(
            user_id=job.user_id,
            document_id=job.document_id,
            tenant_id=job.tenant_id,
        )
        if isinstance(status_snapshot, Success):
            record = status_snapshot.unwrap()
            if isinstance(record, dict):
                raw_metadata = record.get("document_metadata")
                # The extraction checkpoint is written before chunking and
                # embedding.  If a later stage caused Celery to retry, the
                # provider result is already represented in Graphiti and a
                # second provider call would both duplicate writes and spend
                # the cumulative budget again.  Only reuse it for the same
                # document version; legacy rows without version metadata are
                # unambiguously the v1 identity.
                raw_checkpoint_version = (
                    raw_metadata.get("knowledge_extraction_document_version")
                    if isinstance(raw_metadata, dict)
                    else None
                )
                try:
                    checkpoint_version = (
                        int(raw_checkpoint_version)
                        if isinstance(raw_checkpoint_version, (int, str))
                        and not isinstance(raw_checkpoint_version, bool)
                        else None
                    )
                except ValueError:
                    checkpoint_version = None
                same_checkpoint_version = checkpoint_version == job.document_version
                raw_budget = (
                    raw_metadata.get("extraction_budget")
                    if isinstance(raw_metadata, dict) and same_checkpoint_version
                    else None
                )
                if isinstance(raw_budget, dict):
                    try:
                        budget = ExtractionBudget.model_validate(raw_budget)
                    except ValueError:
                        logger.bind(document_id=job.document_id).warning(
                            "document_extraction_budget_checkpoint_invalid"
                        )
                if (
                    isinstance(raw_metadata, dict)
                    and raw_metadata.get("knowledge_extraction") == "complete"
                    and same_checkpoint_version
                ):
                    raw_classes = raw_metadata.get("knowledge_extraction_classes", ())
                    classes = (
                        tuple(value for value in raw_classes if isinstance(value, str))
                        if isinstance(raw_classes, list)
                        else ()
                    )
                    # Restore the relationship-phase truth persisted with the
                    # same checkpoint instead of overwriting it with a fresh
                    # not-attempted state.
                    restored_relationships = _relationship_state_from_metadata(raw_metadata)
                    if restored_relationships is None:
                        restored_relationships = GraphRelationshipsState(
                            status=GraphRelationshipsStatus.NOT_ATTEMPTED,
                            reason="knowledge_extraction_checkpoint_reused",
                        )
                    return (
                        ExtractionSucceeded(
                            documents=(),
                            budget=budget,
                            extraction_classes=classes,
                        ),
                        restored_relationships,
                    )
    # A retry starts a fresh wall-clock window, but its ledger is cumulative.
    # Keeping this transition here (once per ingestion attempt, before the
    # contract loop) prevents a multi-contract document from resetting timing
    # between contracts while avoiding an old attempt's elapsed time blocking
    # every retry immediately.
    budget = budget.model_copy(update={"started_at": datetime.now(tz=UTC)})
    if runtime.extraction is None:
        return (
            ExtractionFailed(
                code=ExtractionFailureCode.UNCONFIGURED,
                message="LangExtract provider is not configured",
                budget=budget,
            ),
            GraphRelationshipsState(
                status=GraphRelationshipsStatus.NOT_ATTEMPTED,
                reason="extraction_unconfigured",
            ),
        )
    documents: list[object] = []
    relationship_states: list[GraphRelationshipsState] = []
    for boundary in boundaries:
        outcome = await runtime.extraction.extract(
            ExtractionRequest(
                text=boundary.markdown,
                prompt_description=_KNOWLEDGE_EXTRACTION_PROMPT,
                document_id=job.document_id,
                tenant_id=job.tenant_id,
                document_version=job.document_version,
                contract_index=boundary.contract_index,
                extraction_passes=settings.LANGEXTRACT_EXTRACTION_PASSES,
                max_workers=settings.LANGEXTRACT_MAX_WORKERS,
                max_char_buffer=settings.LANGEXTRACT_MAX_CHAR_BUFFER,
                budget=budget,
            )
        )
        if isinstance(outcome, ExtractionFailed):
            # Aggregate, never replace: boundaries processed before the failure
            # already produced relationship-phase states that must survive.
            return (
                outcome,
                aggregate_relationship_states(
                    [
                        *relationship_states,
                        GraphRelationshipsState(
                            status=GraphRelationshipsStatus.NOT_ATTEMPTED,
                            reason=f"knowledge_extraction_failed:{outcome.code.value}",
                        ),
                    ]
                ),
            )
        budget = outcome.budget
        documents.extend(outcome.documents)
        segments = _extract_clause_segments(
            outcome,
            document_id=job.document_id,
            contract_index=boundary.contract_index,
        )
        # Second pass: a different prompt over different input. Pass-1 clause
        # identities are fed back as server-derived inventory; document text
        # travels only inside the delimited untrusted region; every edge is
        # span-validated and identity-validated before it can be persisted.
        writer_ready = runtime.graph_writer is not None and runtime.idempotency is not None
        if segments and writer_ready:
            relationship_pass = await run_relationship_pass(
                extraction=runtime.extraction,
                boundary_text=boundary.markdown,
                inventory=[
                    ClauseIdentity(
                        clause_id=segment.clause_id,
                        clause_type=segment.clause_type.value,
                        section_ref=segment.section_ref,
                        char_start=segment.start_char,
                        char_end=segment.end_char,
                    )
                    for segment in segments
                ],
                entity_corpus=[segment.text for segment in segments],
                budget=budget,
                config=relationship_config,
                document_id=job.document_id,
                tenant_id=job.tenant_id,
                document_version=job.document_version,
                contract_index=boundary.contract_index,
                max_char_buffer=settings.LANGEXTRACT_MAX_CHAR_BUFFER,
                max_workers=settings.LANGEXTRACT_MAX_WORKERS,
            )
        else:
            relationship_pass = RelationshipPassResult(
                ran=False,
                skip_reason=(
                    "no_clause_segments_extracted" if not segments else "graph_writer_unconfigured"
                ),
                budget=budget,
            )
        budget = relationship_pass.budget
        legal_edges = [
            _legal_relationship_from_grounded(
                edge,
                document_id=job.document_id,
                contract_index=boundary.contract_index,
            )
            for edge in relationship_pass.edges
        ]
        edge_spans = _absolute_edge_spans(boundary, relationship_pass.edges, legal_edges)
        phase_result = await _write_extracted_clause_episodes(
            segments=segments,
            relationships=legal_edges,
            edge_spans=edge_spans,
            relationship_pass=relationship_pass,
            graph_writer=runtime.graph_writer,
            idempotency=runtime.idempotency,
            document_id=job.document_id,
            tenant_id=job.tenant_id,
            document_version=job.document_version,
            contract_index=boundary.contract_index,
            user_id=job.user_id,
            jurisdiction=jurisdiction,
            document_type=document_type,
        )
        relationship_states.append(phase_result.relationships)
        if not phase_result.success:
            return (
                ExtractionFailed(
                    code=ExtractionFailureCode.GRAPH_WRITE_ERROR,
                    message="One or more extracted clauses could not be persisted to Graphiti",
                    budget=budget,
                ),
                aggregate_relationship_states(relationship_states),
            )
    return (
        ExtractionSucceeded(documents=tuple(documents), budget=budget),
        aggregate_relationship_states(relationship_states),
    )


def _extract_clause_segments(
    outcome: ExtractionSucceeded,
    *,
    document_id: str,
    contract_index: int = 0,
) -> list[ClauseSegment]:
    """Map grounded pass-1 clause extractions into canonical clause segments.

    Ungrounded extractions (``char_interval is None``) are skipped exactly as
    ``langextract_to_graph`` does — they can never become segments or edges.

    Clause identity is issued SERVER-SIDE from the boundary coordinates and
    extraction span, so model-derived strings never become identity and never
    enter the pass-2 prompt's trusted section unvalidated.  The model's
    ``clause_id``/``section_ref`` attribution is mapped onto the issued
    identity as a format-validated section reference only.
    """
    segments: list[ClauseSegment] = []
    valid_types = {member.value: member for member in ClauseType}
    for document in outcome.documents:
        for extraction in document.extractions:
            interval = extraction.char_interval
            if interval is None or interval.start_pos is None or interval.end_pos is None:
                continue
            attributes = extraction.attributes or {}
            raw_type = str(attributes.get("clause_type") or extraction.extraction_class).casefold()
            clause_type = valid_types.get(raw_type, ClauseType.OTHER)
            issued_id = hashlib.sha256(
                f"{document_id}:{contract_index}:{interval.start_pos}:{interval.end_pos}:{extraction.extraction_text}".encode()
            ).hexdigest()[:20]
            section_ref = (
                prompt_safe_attribution(attributes.get("section_ref"))
                or prompt_safe_attribution(
                    attributes.get("clause_id") or attributes.get("clause_number")
                )
                or issued_id
            )
            segments.append(
                ClauseSegment(
                    clause_id=issued_id,
                    clause_type=clause_type,
                    text=extraction.extraction_text,
                    section_ref=section_ref,
                    start_char=interval.start_pos,
                    end_char=interval.end_pos,
                    contract_index=contract_index,
                )
            )
    return segments


def _absolute_edge_spans(
    boundary: ContractBoundary,
    grounded_edges: Sequence[GroundedRelationship],
    legal_edges: Sequence[LegalRelationship],
) -> dict[str, tuple[int, int]]:
    """Map edge ids to document-absolute char spans.

    Pass-2 validates spans in boundary-local coordinates (the text the model
    actually saw), but chunk source spans — and therefore citation
    resolution — are document-absolute.  Persisted edge spans get the
    boundary offset added so both coordinate spaces agree; the deterministic
    edge id keeps hashing the validated boundary-local span, which stays
    stable per contract.
    """
    spans: dict[str, tuple[int, int]] = {}
    for index, relationship in enumerate(legal_edges):
        edge = grounded_edges[index]
        if edge.char_start is None or edge.char_end is None:
            continue
        spans[relationship.edge_id] = (
            boundary.start_char + edge.char_start,
            boundary.start_char + edge.char_end,
        )
    return spans


def _legal_relationship_from_grounded(
    edge: GroundedRelationship,
    *,
    document_id: str,
    contract_index: int,
) -> LegalRelationship:
    """Convert a validated pass-2 edge into the canonical writer edge model.

    Server-owned identity: only ``document_id``/``contract_index`` (job
    identity) and content-derived entity/clause text enter the edge.  Content
    can never influence tenant, group, or writer binding — the Graphiti
    writer derives those from the server context, and the injection gate
    proves it for this pass too.
    """
    return LegalRelationship(
        edge_id=relationship_edge_id(
            document_id=document_id,
            contract_index=contract_index,
            edge=edge,
        ),
        from_node=edge.from_entity,
        relationship=edge.relationship,
        to_node=edge.to_entity,
        clause_id=edge.clause_id,
        citation=Citation(
            claim=f"{edge.from_entity} {edge.relationship.value} {edge.to_entity}",
            source=f"{document_id}#contract={contract_index};clause={edge.clause_id}",
            confidence=edge.confidence,
            quoted_text=edge.quote[:2_000],
        ),
        contract_index=contract_index,
    )


def _relationship_state_from_pass(
    *,
    relationship_pass: RelationshipPassResult,
    relationship_results: Sequence[object],
) -> GraphRelationshipsState:
    """Typed truth for the relationship phase given the pass and write results."""
    attempted = len(relationship_results)
    succeeded = sum(1 for result in relationship_results if getattr(result, "success", False))
    # A provider failure returns ran=True WITH provider_error and zero edges.
    # It must surface as a failed phase BEFORE the ran/ran=False branching —
    # otherwise an outage persists as a successful pass with zero edges.
    if relationship_pass.provider_error is not None:
        return GraphRelationshipsState(
            status=GraphRelationshipsStatus.FAILED,
            reason=f"relationship_provider_error:{relationship_pass.provider_error}",
            relationships_attempted=attempted,
            relationships_succeeded=succeeded,
            relationships_invalid_span=relationship_pass.drops.invalid_span,
            relationships_dropped=relationship_pass.drops.total
            - relationship_pass.drops.invalid_span,
        )
    if not relationship_pass.ran:
        if relationship_pass.budget_exhausted:
            skip_reason = RELATIONSHIP_BUDGET_EXHAUSTED_REASON
        else:
            skip_reason = relationship_pass.skip_reason or "relationship_pass_not_run"
        return GraphRelationshipsState(
            status=(
                GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED
                if skip_reason == RELATIONSHIP_CONFIG_DISABLED_REASON
                else GraphRelationshipsStatus.NOT_ATTEMPTED
            ),
            reason=skip_reason,
            relationships_dropped=relationship_pass.drops.total,
        )
    write_failures = attempted - succeeded
    return GraphRelationshipsState(
        status=(
            GraphRelationshipsStatus.FAILED
            if write_failures
            else GraphRelationshipsStatus.SUCCEEDED
        ),
        reason="relationship_edge_write_failed" if write_failures else None,
        relationships_attempted=attempted,
        relationships_succeeded=succeeded,
        relationships_invalid_span=relationship_pass.drops.invalid_span,
        relationships_dropped=relationship_pass.drops.total - relationship_pass.drops.invalid_span,
    )


async def _write_extracted_clause_episodes(
    *,
    segments: list[ClauseSegment],
    relationships: list[LegalRelationship],
    edge_spans: dict[str, tuple[int, int]],
    relationship_pass: RelationshipPassResult,
    graph_writer: object | None,
    idempotency: object | None,
    document_id: str,
    tenant_id: str = "",
    document_version: int = 1,
    contract_index: int = 0,
    user_id: str,
    jurisdiction: str | None,
    document_type: str,
) -> GraphWritePhaseResult:
    """Map grounded clause extractions into the existing canonical writer.

    Success is computed over clause writes only: a clause-write failure fails
    the phase (the document degrades), while the relationship phase reports
    its own typed truth — extraction outcome, drop counts, and per-edge write
    results — through ``GraphRelationshipsState``.
    """
    if not segments:
        return GraphWritePhaseResult(
            success=True,
            relationships=GraphRelationshipsState(
                status=GraphRelationshipsStatus.NOT_ATTEMPTED,
                reason="no_clause_segments_extracted",
                relationships_dropped=relationship_pass.drops.total,
            ),
        )
    if graph_writer is None and idempotency is None:
        return GraphWritePhaseResult(
            success=True,
            clause_writes_attempted=0,
            clause_writes_succeeded=0,
            relationships=GraphRelationshipsState(
                status=GraphRelationshipsStatus.NOT_ATTEMPTED,
                reason="graph_writer_unconfigured",
                relationships_dropped=relationship_pass.drops.total,
            ),
        )
    if graph_writer is None or idempotency is None:
        logger.bind(
            graph_writer_configured=graph_writer is not None,
            idempotency_configured=idempotency is not None,
        ).error("graph_clause_writer_dependencies_incomplete")
        return GraphWritePhaseResult(
            success=False,
            relationships=GraphRelationshipsState(
                status=GraphRelationshipsStatus.NOT_ATTEMPTED,
                reason="graph_writer_dependencies_incomplete",
                relationships_dropped=relationship_pass.drops.total,
            ),
        )
    clause_results, relationship_results = await write_clause_episodes_to_graphiti(
        segments,
        relationships,
        GraphIngestionContext(
            document_id=document_id,
            tenant_id=tenant_id,
            document_version=document_version,
            contract_index=contract_index,
            user_id=user_id,
            thread_id=f"ingestion:{document_id}",
            jurisdiction=jurisdiction or "unspecified",
            document_type=document_type,
        ),
        graphiti_service=cast("ClauseWriterService", graph_writer),
        idempotency=cast("IdempotencyGuard", idempotency),
        edge_spans=edge_spans or None,
    )
    relationships_state = _relationship_state_from_pass(
        relationship_pass=relationship_pass,
        relationship_results=relationship_results,
    )
    log = logger.bind(
        document_id=document_id,
        relationships_attempted=relationships_state.relationships_attempted,
        relationships_succeeded=relationships_state.relationships_succeeded,
        relationships_invalid_span=relationships_state.relationships_invalid_span,
        relationships_dropped=relationships_state.relationships_dropped,
    )
    if relationships_state.status is GraphRelationshipsStatus.FAILED:
        log.warning("graph_relationships_failed", reason=relationships_state.reason)
    elif relationships_state.status is GraphRelationshipsStatus.SKIPPED_NOT_IMPLEMENTED:
        log.warning("graph_relationships_skipped", reason=relationships_state.reason)
    else:
        log.info("graph_relationships_written")
    return GraphWritePhaseResult(
        success=all(result.success for result in clause_results),
        clause_writes_attempted=len(clause_results),
        clause_writes_succeeded=sum(1 for result in clause_results if result.success),
        relationships=relationships_state,
    )


async def _verify_legal_chunks(
    *,
    repo: DocumentRepository,
    graphiti: Graphiti | None,
    user_id: str,
    tenant_id: str,
    document_id: str,
    chunk_rows: list[dict[str, object]],
) -> DocumentResult[int]:
    verified_count = 0
    for chunk in chunk_rows:
        raw_contract_index = chunk.get("contract_index")
        contract_index = (
            int(raw_contract_index)
            if isinstance(raw_contract_index, int) and not isinstance(raw_contract_index, bool)
            else None
        )
        raw_document_version = chunk.get("document_version", 1)
        document_version = (
            int(raw_document_version)
            if isinstance(raw_document_version, int) and not isinstance(raw_document_version, bool)
            else 1
        )
        result = await write_and_verify_chunk(
            graphiti=graphiti,
            user_id=user_id,
            document_id=document_id,
            chunk_id=str(chunk["id"]),
            clause_type=str(chunk["clause_type"]) if chunk.get("clause_type") is not None else None,
            preamble=str(chunk.get("preamble", "")),
            content=str(chunk["content"]),
            tenant_id=tenant_id,
            contract_index=contract_index,
            document_version=document_version,
        )
        chunk["graphiti_episode_id"] = result.episode_id
        chunk["graphiti_verified"] = result.verified
        if result.verified:
            verified_count += 1
    upsert_result = await repo.upsert_chunks(
        build_chunk_rows(
            document_id=document_id,
            user_id=user_id,
            tenant_id=tenant_id,
            chunks=chunk_rows,
        )
    )
    if isinstance(upsert_result, Failure):
        log_expected_failure(upsert_result.failure(), operation="verify_legal_chunks")
        return Failure(upsert_result.failure())
    return Success(verified_count)
