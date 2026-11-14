"""Document ingestion graph wrapper."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypedDict, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from returns.result import Failure

from app.ai.embeddings import EmbeddingQuotaExhaustedError
from app.events import with_terminal_outbox
from app.features.documents.api.http.dto import IngestionJob, IngestionRuntime
from app.kernel.result import log_expected_failure
from app.observability.logging import logger
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from typing import Any

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig

    from app.extraction.langextract.service import AsyncExtractionService
    from app.features.documents.adapters.persistence.postgres.document_repository import (
        DocumentRepository,
    )
    from app.features.documents.domain.errors import DocumentResult
    from app.utils.storage import StorageService

type IngestDocumentFn = Callable[..., Awaitable[DocumentResult[dict[str, object]]]]

_REPOSITORY_CONFIG_KEY = "document_repository"
_IDEMPOTENCY_CONFIG_KEY = "document_idempotency"


def get_document_repository(config: RunnableConfig) -> DocumentRepository:
    """Resolve the job-scoped repository from LangGraph invocation wiring."""
    repository = config.get("configurable", {}).get(_REPOSITORY_CONFIG_KEY)
    if repository is None:
        raise ServiceUnavailableException(
            detail="Document ingestion collaborator is unavailable",
            data={"collaborator": _REPOSITORY_CONFIG_KEY},
        )
    return cast("DocumentRepository", repository)


class DocumentIngestionState(TypedDict, total=False):
    """Per-job ingestion state: plain-data channels, natively checkpointer-safe."""

    document_id: str
    user_id: str
    tenant_id: str
    filename: str
    content_type: str
    object_uri: str
    document_version: int
    attempt: int
    task_id: str
    status: str
    chunk_count: int
    verified_chunk_count: int
    document_kind: str
    error_code: str
    error_message: str
    error_retryable: bool


def build_document_ingestion_graph(
    *,
    object_store: StorageService,
    graphiti: Graphiti | None,
    ingest_document_fn: IngestDocumentFn,
    llm: BaseChatModel,
    extraction: AsyncExtractionService | None = None,
    graph_writer: object | None = None,
    idempotency: object | None = None,
) -> CompiledStateGraph[Any]:
    """Build the process-scoped ingestion graph without job-scoped collaborators."""

    graph = StateGraph(DocumentIngestionState)  # ty: ignore[invalid-argument-type] - stub bound is imprecise for TypedDicts; same ignore as retrieval_kb/graph.py
    graph.add_node(
        "ingest_document",
        cast(
            "Any",
            _make_ingest_document_node(
                object_store=object_store,
                graphiti=graphiti,
                ingest_document_fn=ingest_document_fn,
                llm=llm,
                extraction=extraction,
                graph_writer=graph_writer,
                idempotency=idempotency,
            ),
        ),
        input_schema=cast("Any", DocumentIngestionState),
    )
    graph.set_entry_point("ingest_document")
    graph.add_edge("ingest_document", END)
    return cast("CompiledStateGraph[Any]", graph.compile())


def _make_ingest_document_node(
    *,
    object_store: StorageService,
    graphiti: Graphiti | None,
    ingest_document_fn: IngestDocumentFn,
    llm: BaseChatModel,
    extraction: AsyncExtractionService | None = None,
    graph_writer: object | None = None,
    idempotency: object | None = None,
) -> Callable[[DocumentIngestionState, RunnableConfig], Awaitable[dict[str, object]]]:
    async def ingest_document_node(
        state: DocumentIngestionState, config: RunnableConfig
    ) -> dict[str, object]:
        repo = get_document_repository(config)
        job_idempotency = config.get("configurable", {}).get(
            _IDEMPOTENCY_CONFIG_KEY,
            idempotency,
        )
        job = IngestionJob(
            document_id=state.get("document_id", ""),
            user_id=state.get("user_id", ""),
            tenant_id=state.get("tenant_id", ""),
            document_version=state.get("document_version", 1),
            attempt=state.get("attempt", 0),
            task_id=_terminal_task_id(config),
            filename=state.get("filename", ""),
            content_type=state.get("content_type", ""),
            object_uri=state.get("object_uri", ""),
        )
        try:
            result = await ingest_document_fn(
                job=job,
                runtime=IngestionRuntime(
                    object_store=object_store,
                    repo=repo,
                    graphiti=graphiti,
                    llm=llm,
                    extraction=extraction,
                    graph_writer=graph_writer,
                    idempotency=job_idempotency,
                    transaction_checkpoint=repo.session.commit,
                ),
            )
        except EmbeddingQuotaExhaustedError as exc:
            # STRIDE D-3: a daily-quota 429 is not an unexpected failure — it
            # must not read as ``INGESTION_UNEXPECTED_FAILURE``, burn the Celery
            # retry budget, or dead-letter. The document parks in the
            # NON-terminal ``quota_exhausted`` status (typed metadata, resumable
            # by the next quota window's idempotent driver) and the task
            # returns cleanly instead of raising.
            exc.add_note(f"document_id={job.document_id}, operation=ingest_document_node")
            logger.bind(document_id=job.document_id, model=exc.model).warning(
                "Document ingestion parked: embedding quota exhausted"
            )
            await _record_quota_exhaustion(
                repo,
                document_id=job.document_id,
                tenant_id=job.tenant_id,
                document_version=job.document_version,
                attempt=job.attempt,
                task_id=job.task_id,
                detail=exc.detail,
            )
            return {
                "status": "quota_exhausted",
                "error_code": EmbeddingQuotaExhaustedError.error_code,
                "error_message": exc.detail,
                "error_retryable": False,
            }
        except Exception as exc:  # noqa: BLE001 — worker boundary must persist failure
            exc.add_note(f"document_id={job.document_id}, operation=ingest_document_node")
            logger.bind(
                document_id=job.document_id,
                operation="ingest_document_node",
                error=str(exc),
            ).exception("Document ingestion failed unexpectedly")
            # Celery owns retryable failures.  Recording ``failed`` here would
            # make the document look terminal during the retry window and
            # would also stage a terminal event that can race the retry.
            return {
                "status": "failed",
                "error_code": "INGESTION_UNEXPECTED_FAILURE",
                "error_message": "Document ingestion failed unexpectedly",
                "error_retryable": True,
            }
        if isinstance(result, Failure):
            error = result.failure()
            log_expected_failure(error=error, operation="ingest_document_node")
            if not error.retryable:
                await _record_ingestion_failure(
                    repo,
                    document_id=job.document_id,
                    user_id=job.user_id,
                    tenant_id=job.tenant_id,
                    document_version=job.document_version,
                    attempt=job.attempt,
                    task_id=job.task_id,
                    error_code=str(error.code),
                    error_message=error.message,
                    retryable=False,
                    details=error.details,
                )
            return {
                "status": "failed",
                "error_code": str(error.code),
                "error_message": error.message,
                "error_retryable": error.retryable,
            }
        return result.unwrap()

    return ingest_document_node


async def _record_quota_exhaustion(
    repo: DocumentRepository,
    *,
    document_id: str,
    tenant_id: str = "",
    document_version: int = 1,
    attempt: int = 0,
    task_id: str | None = None,
    detail: str,
) -> None:
    """Persist the non-terminal ``quota_exhausted`` park with typed metadata.

    Deliberately unlike ``_record_ingestion_failure``: no terminal ``failed``
    status, no terminal outbox event — the document is parked, not dead. The
    metadata rides the same typed ``ingestion_error`` shape so the cause is
    one read away, and ``extraction_incomplete`` stays ``False`` (extraction
    succeeded; the failure is at the indexing seam).
    """
    metadata: dict[str, object] = {
        "ingestion_error": {
            "code": EmbeddingQuotaExhaustedError.error_code,
            "message": detail,
            "retryable": False,
            "ingestion_state": {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "document_version": document_version,
                "attempt": attempt,
                "task_id": task_id or "",
            },
        }
    }
    try:
        status_result = await repo.update_document_status(
            document_id=document_id,
            status="quota_exhausted",
            metadata_=metadata,
        )
    except Exception as exc:  # noqa: BLE001 — park persistence must not mask the task outcome
        exc.add_note(f"document_id={document_id}, operation=record_quota_exhaustion")
        logger.bind(document_id=document_id, error=str(exc)).exception(
            "Quota-exhausted status persistence raised"
        )
    else:
        if isinstance(status_result, Failure):
            log_expected_failure(
                error=status_result.failure(), operation="record_quota_exhaustion"
            )


async def _record_ingestion_failure(
    repo: DocumentRepository,
    *,
    document_id: str,
    user_id: str = "",
    tenant_id: str = "",
    document_version: int = 1,
    attempt: int = 0,
    task_id: str | None = None,
    error_code: str,
    error_message: str,
    retryable: bool,
    details: dict[str, object] | None = None,
) -> None:
    """Persist one exhausted/non-retryable failure and its terminal event."""
    if retryable:
        return
    metadata: dict[str, object] = {
        "ingestion_error": {
            "code": error_code,
            "message": error_message,
            "retryable": retryable,
            "ingestion_state": {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "document_version": document_version,
                "attempt": attempt,
                "task_id": task_id or "",
            },
            **(details or {}),
        }
    }
    if details is not None and "extraction_budget" in details:
        metadata["extraction_budget"] = details["extraction_budget"]
    if details is not None and "knowledge_extraction_failure" in details:
        metadata["knowledge_extraction_failure"] = details["knowledge_extraction_failure"]
    try:
        status_result = await repo.update_document_status(
            document_id=document_id,
            status="failed",
            metadata_=metadata,
            extraction_incomplete=True,
        )
    except Exception as exc:  # noqa: BLE001 — failure recording must not mask terminal state
        exc.add_note(f"document_id={document_id}, operation=record_ingestion_failure_status")
        logger.bind(document_id=document_id, error=str(exc)).exception(
            "Document failure status persistence raised"
        )
    else:
        if isinstance(status_result, Failure):
            log_expected_failure(
                error=status_result.failure(), operation="record_ingestion_failure"
            )
    session = getattr(repo, "session", None)
    if session is not None:
        try:
            await with_terminal_outbox(
                session,
                document_id=document_id,
                payload={
                    "document_id": document_id,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "status": "failed",
                    "document_version": document_version,
                    "attempt": attempt,
                    "task_id": task_id,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
        except Exception as exc:  # noqa: BLE001 — outbox degradation is audited, not fatal
            exc.add_note(f"document_id={document_id}, operation=record_ingestion_failure_outbox")
            logger.bind(document_id=document_id, error=str(exc)).exception(
                "Document terminal outbox staging failed"
            )


def _terminal_task_id(config: RunnableConfig) -> str | None:
    """Read worker identity from invocation metadata without trusting payload data."""
    configurable = config.get("configurable", {})
    raw = configurable.get("task_id") or configurable.get("celery_task_id")
    return raw if isinstance(raw, str) and raw else None
