"""Document ingestion graph wrapper."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypedDict, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from returns.result import Failure

from app.kernel.result import log_expected_failure
from app.observability.logging import logger
from app.utils import ServiceUnavailableException

from .dto import IngestionJob, IngestionRuntime

if TYPE_CHECKING:
    from typing import Any

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig

    from app.connections.object_storage import StorageService
    from app.extraction.langextract.service import AsyncExtractionService

    from .exceptions import DocumentResult
    from .repository import DocumentRepository

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
        except Exception as exc:  # noqa: BLE001 — worker boundary must persist failure
            exc.add_note(f"document_id={job.document_id}, operation=ingest_document_node")
            logger.bind(
                document_id=job.document_id,
                operation="ingest_document_node",
                error=str(exc),
            ).exception("Document ingestion failed unexpectedly")
            await _record_ingestion_failure(
                repo,
                document_id=job.document_id,
                error_code="INGESTION_UNEXPECTED_FAILURE",
                error_message="Document ingestion failed unexpectedly",
                retryable=True,
            )
            return {
                "status": "failed",
                "error_code": "INGESTION_UNEXPECTED_FAILURE",
                "error_message": "Document ingestion failed unexpectedly",
                "error_retryable": True,
            }
        if isinstance(result, Failure):
            error = result.failure()
            log_expected_failure(error=error, operation="ingest_document_node")
            await _record_ingestion_failure(
                repo,
                document_id=job.document_id,
                error_code=str(error.code),
                error_message=error.message,
                retryable=error.retryable,
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


async def _record_ingestion_failure(
    repo: DocumentRepository,
    *,
    document_id: str,
    error_code: str,
    error_message: str,
    retryable: bool,
    details: dict[str, object] | None = None,
) -> None:
    """Persist a terminal/early failure so ``received`` cannot lie forever."""
    metadata: dict[str, object] = {
        "ingestion_error": {
            "code": error_code,
            "message": error_message,
            "retryable": retryable,
            **(details or {}),
        }
    }
    status_result = await repo.update_document_status(
        document_id=document_id,
        status="failed",
        metadata_=metadata,
        extraction_incomplete=True,
    )
    if isinstance(status_result, Failure):
        log_expected_failure(error=status_result.failure(), operation="record_ingestion_failure")
