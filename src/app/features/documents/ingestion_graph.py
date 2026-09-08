"""Document ingestion graph wrapper."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypedDict, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from returns.result import Failure

from app.shared.result import log_expected_failure

from .dto import IngestionJob, IngestionRuntime

if TYPE_CHECKING:
    from typing import Any

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel

    from app.shared.services.storage import StorageService

    from .errors import DocumentResult
    from .repository import DocumentRepository

type IngestDocumentFn = Callable[..., Awaitable[DocumentResult[dict[str, object]]]]


class DocumentIngestionState(TypedDict, total=False):
    """Per-job ingestion state: plain-data channels, natively checkpointer-safe."""

    document_id: str
    user_id: str
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
    repo: DocumentRepository,
    graphiti: Graphiti | None,
    ingest_document_fn: IngestDocumentFn,
    llm: BaseChatModel,
) -> CompiledStateGraph[Any]:
    """Build the per-job ingestion graph."""

    graph = StateGraph(DocumentIngestionState)  # ty: ignore[invalid-argument-type] - stub bound is imprecise for TypedDicts; same ignore as retrieval_kb/graph.py
    graph.add_node(
        "ingest_document",
        cast(
            "Any",
            _make_ingest_document_node(
                object_store=object_store,
                repo=repo,
                graphiti=graphiti,
                ingest_document_fn=ingest_document_fn,
                llm=llm,
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
    repo: DocumentRepository,
    graphiti: Graphiti | None,
    ingest_document_fn: IngestDocumentFn,
    llm: BaseChatModel,
) -> Callable[[DocumentIngestionState], Awaitable[dict[str, object]]]:
    async def ingest_document_node(state: DocumentIngestionState) -> dict[str, object]:
        result = await ingest_document_fn(
            job=IngestionJob(
                document_id=state.get("document_id", ""),
                user_id=state.get("user_id", ""),
                filename=state.get("filename", ""),
                content_type=state.get("content_type", ""),
                object_uri=state.get("object_uri", ""),
            ),
            runtime=IngestionRuntime(
                object_store=object_store,
                repo=repo,
                graphiti=graphiti,
                llm=llm,
            ),
        )
        if isinstance(result, Failure):
            error = result.failure()
            log_expected_failure(error=error, operation="ingest_document_node")
            return {
                "status": "failed",
                "error_code": str(error.code),
                "error_message": error.message,
                "error_retryable": error.retryable,
            }
        return result.unwrap()

    return ingest_document_node
