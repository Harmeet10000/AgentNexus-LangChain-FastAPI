"""Document ingestion graph wrapper."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from typing import Any

    from graphiti_core.graphiti import Graphiti
    from langchain_core.language_models import BaseChatModel

    from app.shared.services.storage import StorageService

    from .repository import DocumentRepository

type IngestDocumentFn = Callable[..., Awaitable[dict[str, object]]]


class DocumentIngestionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = ""
    user_id: str = ""
    filename: str = ""
    content_type: str = ""
    object_uri: str = ""
    status: str = ""
    chunk_count: int = 0
    verified_chunk_count: int = 0
    document_kind: str = ""


def build_document_ingestion_graph(
    *,
    object_store: StorageService,
    repo: DocumentRepository,
    graphiti: Graphiti | None,
    ingest_document_fn: IngestDocumentFn,
    llm: BaseChatModel,
) -> CompiledStateGraph[Any]:
    """Build the per-job ingestion graph."""

    graph = StateGraph(DocumentIngestionState)
    graph.add_node(
        node="ingest_document",
        action=_make_ingest_document_node(
                object_store=object_store,
                repo=repo,
                graphiti=graphiti,
                ingest_document_fn=ingest_document_fn,
                llm=llm,
            ),
    )
    graph.set_entry_point("ingest_document")
    graph.add_edge("ingest_document", END)
    return graph.compile()


def _make_ingest_document_node(
    *,
    object_store: StorageService,
    repo: DocumentRepository,
    graphiti: Graphiti | None,
    ingest_document_fn: IngestDocumentFn,
    llm: BaseChatModel,
) -> IngestDocumentFn:
    async def ingest_document_node(state: DocumentIngestionState) -> dict[str, object]:
        return await ingest_document_fn(
            document_id=state.document_id,
            user_id=state.user_id,
            filename=state.filename,
            content_type=state.content_type,
            object_uri=state.object_uri,
            object_store=object_store,
            repo=repo,
            graphiti=graphiti,
            llm=llm,
        )

    return ingest_document_node
