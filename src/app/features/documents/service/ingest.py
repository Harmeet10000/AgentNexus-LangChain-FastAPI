"""Unified document feature services."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from returns.result import Failure, Success
from sqlalchemy import text

from app.ai.embeddings import EmbeddingTaskType, embed_texts
from app.config import get_settings
from app.connections.object_storage import build_s3_key, key_from_s3_uri
from app.extraction.langextract.service import (
    ExtractionFailed,
    ExtractionFailureCode,
    ExtractionRequest,
    ExtractionSucceeded,
)
from app.features.auth.tenancy import personal_tenant_id
from app.features.documents.access_audit import (
    DocumentAccessAuditEvent,
    DocumentAccessAuditRecorder,
    InMemoryDocumentAccessAuditRecorder,
)
from app.features.documents.classification import (
    ClassifiedDocument,
    ContractBoundary,
    ParsedDocument,
    QualityWarning,
    classify_document,
    establish_contract_boundaries,
    segment_chunks,
)
from app.features.documents.constants import (
    ANALYZE_THRESHOLD_CHUNKS,
    INGEST_EMBEDDING_BATCH_SIZE,
)
from app.features.documents.dto import (
    DocumentStatusResponse,
    DocumentUploadResponse,
    QualityWarningDTO,
)
from app.features.documents.exceptions import (
    DocumentGraphWriteError,
    DocumentIngestionCheckpointError,
    DocumentNotFoundError,
    DocumentStorageError,
    DocumentValidationError,
)
from app.features.documents.graphiti_verifier import write_and_verify_chunk
from app.features.documents.legal_metadata import (
    contract_event_dates,
    enrich_legal_chunks,
    extract_legal_metadata,
)
from app.features.documents.parser import parse_document
from app.features.documents.repository import (
    DocumentRepository,
    build_chunk_rows,
)
from app.kernel.result import log_expected_failure
from app.observability.logging import logger, trace_layer
from app.shared.jobs.names import DOCUMENTS_INGEST

if TYPE_CHECKING:
    from typing import Any, Literal

    from graphiti_core.graphiti import Graphiti
    from langgraph.graph.state import CompiledStateGraph
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, async_sessionmaker

    from app.connections.object_storage import StorageService
    from app.features.documents.classification import PreparedChunk
    from app.features.documents.dto import IngestionJob, IngestionRuntime
    from app.features.documents.exceptions import DocumentResult
    from app.features.documents.legal_metadata import (
        LegalMetadataExtraction,
    )
    from app.resilience.idempotency import IdempotencyGuard
    from app.retrieval.graphiti.write_clause_episodes import (
        GraphitiService as ClauseWriterService,
    )

from app.features.documents.service.query import (
    _KNOWLEDGE_EXTRACTION_PROMPT,
    _batched,
    _flatten_warnings,
    _LegalPreparation,
)


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

        from app.events import (
            with_outbox,
        )

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
    legal = legal_preparations[0][1] if len(legal_preparations) == 1 else _LegalPreparation(
        metadata=None,
        warnings=[warning for _, preparation in legal_preparations for warning in preparation.warnings],
    )
    extraction_outcome = await _extract_document_knowledge(
        runtime=runtime,
        job=job,
        markdown=parsed.markdown,
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
        parties=[*(legal.metadata.parties if legal.metadata else (classified.parties if len(boundaries) == 1 else []))],
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
                details={"document_id": job.document_id},
                source="graphiti",
            )
        )
    checkpoint_result = await _checkpoint_ingestion_transaction(runtime)
    if isinstance(checkpoint_result, Failure):
        return Failure(checkpoint_result.failure())
    # Chunking happens only after extraction/status checkpoint.  This preserves
    # the old transactional ordering while keeping each contract isolated.
    for index, (boundary, contract_parsed, contract_classified) in enumerate(
        contract_parsed_items
    ):
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
        prepared_chunks.extend(_annotate_contract_chunks(contract_chunks, boundary))
        segmentation_warnings.extend(warnings)

    # Chunk identity remains unique within the document while contract_index
    # preserves the independent legal identity used by evidence citations.
    prepared_chunks = [
        chunk.model_copy(update={"chunk_index": index})
        for index, chunk in enumerate(prepared_chunks)
    ]
    chunk_rows = await _embed_chunks(
        user_id=job.user_id,
        document_id=job.document_id,
        chunks=prepared_chunks,
        extra_warnings=(
            boundary_result.warnings
            + segmentation_warnings
            + classified.warnings
            + legal.warnings
        ),
        extraction_outcome=extraction_outcome,
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
                runtime.graphiti,
                preparation.metadata,
                job.document_id,
                contract_index=boundary.contract_index,
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
            "completed" if verified_count == len(chunk_rows) else "completed_with_warnings"
        )
    else:
        final_status = (
            "completed"
            if not segmentation_warnings and not classified.warnings
            else "completed_with_warnings"
        )
        verified_count = 0
    status_result = await runtime.repo.update_document_status(
        document_id=job.document_id, status=final_status
    )
    if isinstance(status_result, Failure):
        return Failure(status_result.failure())
    return Success(
        {
            "status": final_status,
            "document_id": job.document_id,
            "chunk_count": len(chunk_rows),
            "verified_chunk_count": verified_count,
            "document_kind": classified.document_kind,
        }
    )


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
        return ClassifiedDocument(
            document_kind="generic", warnings=list(boundary_warnings)
        )
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
        graphiti_required=any(classification.graphiti_required for classification in classifications),
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
    document_id: str,
    contract_index: int | None = None,
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
                group_id=document_id,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.bind(
                document_id=document_id,
                event_name=event_name,
                event_date=event_date,
            ).warning("graphiti_event_episode_failed", error=str(exc))


@trace_layer("service")
async def run_document_ingestion_task(
    *,
    document_id: str,
    user_id: str,
    tenant_id: str | None = None,
    filename: str,
    content_type: str,
    object_uri: str,
    graph: CompiledStateGraph[Any],
    engine: AsyncEngine,
    session_local: async_sessionmaker[Any],
    idempotency: IdempotencyGuard,
) -> dict[str, object]:
    tenant_id = tenant_id or personal_tenant_id(user_id)
    lock_key = _document_ingestion_lock_key(tenant_id, document_id)
    async with engine.connect() as lock_connection:
        acquired = await lock_connection.scalar(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": lock_key},
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
                result = await graph.ainvoke(
                    {
                        "document_id": document_id,
                        "user_id": user_id,
                        "tenant_id": tenant_id,
                        "filename": filename,
                        "content_type": content_type,
                        "object_uri": object_uri,
                    },
                    {
                        "configurable": {
                            "document_repository": repo,
                            "document_idempotency": job_idempotency,
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
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": lock_key},
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


def _document_ingestion_lock_key(user_id: str, document_id: str) -> int:
    """Build a stable signed-bigint key for PostgreSQL advisory locking."""
    identity = f"{user_id}\0{document_id}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], byteorder="big", signed=True)


async def _embed_chunks(
    *,
    user_id: str,
    document_id: str,
    chunks: list[PreparedChunk],
    extra_warnings: list[QualityWarning],
    extraction_outcome: ExtractionSucceeded | ExtractionFailed,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch in _batched(chunks, INGEST_EMBEDDING_BATCH_SIZE):
        embeddings = await embed_texts(
            [f"{chunk.preamble}\n\n{chunk.content}".strip() for chunk in batch],
            task_type=EmbeddingTaskType.DOCUMENT,
        )
        for chunk, embedding in zip(batch, embeddings, strict=True):
            extraction_metadata = _extraction_metadata(extraction_outcome)
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
) -> dict[str, object]:
    if isinstance(outcome, ExtractionFailed):
        return {
            "knowledge_extraction": "incomplete",
            "knowledge_extraction_failure": str(outcome.code),
        }
    classes = sorted(
        {
            extraction.extraction_class
            for document in outcome.documents
            for extraction in document.extractions
        }
    )
    return {
        "knowledge_extraction": "complete",
        "knowledge_extraction_classes": classes,
    }


def _is_graph_write_failure(outcome: ExtractionSucceeded | ExtractionFailed) -> bool:
    return (
        isinstance(outcome, ExtractionFailed)
        and outcome.code is ExtractionFailureCode.GRAPH_WRITE_ERROR
    )


async def _extract_document_knowledge(
    *,
    runtime: IngestionRuntime,
    job: IngestionJob,
    markdown: str,
    jurisdiction: str | None,
    document_type: str,
) -> ExtractionSucceeded | ExtractionFailed:
    if runtime.extraction is None:
        return ExtractionFailed(
            code=ExtractionFailureCode.UNCONFIGURED,
            message="LangExtract provider is not configured",
        )
    outcome = await runtime.extraction.extract(
        ExtractionRequest(text=markdown, prompt_description=_KNOWLEDGE_EXTRACTION_PROMPT)
    )
    if isinstance(outcome, ExtractionSucceeded):
        graph_write_succeeded = await _write_extracted_clause_episodes(
            outcome=outcome,
            graph_writer=runtime.graph_writer,
            idempotency=runtime.idempotency,
            document_id=job.document_id,
            user_id=job.user_id,
            jurisdiction=jurisdiction,
            document_type=document_type,
        )
        if not graph_write_succeeded:
            return ExtractionFailed(
                code=ExtractionFailureCode.GRAPH_WRITE_ERROR,
                message="One or more extracted clauses could not be persisted to Graphiti",
            )
    return outcome


async def _write_extracted_clause_episodes(
    *,
    outcome: ExtractionSucceeded,
    graph_writer: object | None,
    idempotency: object | None,
    document_id: str,
    user_id: str,
    jurisdiction: str | None,
    document_type: str,
) -> bool:
    """Map grounded clause extractions into the existing canonical writer."""
    from app.extraction.langextract.langextract_to_graph import GraphIngestionContext
    from app.features.agent_saul.graph.state import ClauseSegment, ClauseType
    from app.retrieval.graphiti.write_clause_episodes import write_clause_episodes_to_graphiti

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
            raw_id = attributes.get("clause_id") or attributes.get("clause_number")
            clause_id = (
                str(raw_id)
                if raw_id
                else hashlib.sha256(
                    f"{document_id}:{interval.start_pos}:{interval.end_pos}:{extraction.extraction_text}".encode()
                ).hexdigest()[:20]
            )
            segments.append(
                ClauseSegment(
                    clause_id=clause_id,
                    clause_type=clause_type,
                    text=extraction.extraction_text,
                    section_ref=str(attributes.get("section_ref") or clause_id),
                    start_char=interval.start_pos,
                    end_char=interval.end_pos,
                )
            )
    if not segments:
        return True
    if graph_writer is None and idempotency is None:
        return True
    if graph_writer is None or idempotency is None:
        logger.bind(
            graph_writer_configured=graph_writer is not None,
            idempotency_configured=idempotency is not None,
        ).error("graph_clause_writer_dependencies_incomplete")
        return False
    clause_results, relationship_results = await write_clause_episodes_to_graphiti(
        segments,
        [],
        GraphIngestionContext(
            document_id=document_id,
            user_id=user_id,
            thread_id=f"ingestion:{document_id}",
            jurisdiction=jurisdiction or "unspecified",
            document_type=document_type,
        ),
        graphiti_service=cast("ClauseWriterService", graph_writer),
        idempotency=cast("IdempotencyGuard", idempotency),
    )
    return all(result.success for result in [*clause_results, *relationship_results])


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
        result = await write_and_verify_chunk(
            graphiti=graphiti,
            user_id=user_id,
            document_id=document_id,
            chunk_id=str(chunk["id"]),
            clause_type=str(chunk["clause_type"]) if chunk.get("clause_type") is not None else None,
            preamble=str(chunk.get("preamble", "")),
            content=str(chunk["content"]),
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
