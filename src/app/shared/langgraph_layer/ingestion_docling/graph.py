from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .pipeline_node import (
    make_embed_store_node,
    make_extract_node,
    make_validate_node,
)
from .state import IngestionState

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncEngine

    from app.shared.rag.graphiti import GraphitiService

    from .state import EmbeddingFunction, ExtractionRunnable


def build_ingestion_graph(
    extraction_llm: ExtractionRunnable,
    db_engine: AsyncEngine,
    embedding_fn: EmbeddingFunction,
    _graphiti_service: GraphitiService,
) -> CompiledStateGraph:
    """Build the ingestion graph once during application startup."""
    graph = StateGraph(IngestionState)
    graph.add_node("extract", cast("Any", make_extract_node(extraction_llm)))
    graph.add_node("validate", cast("Any", make_validate_node()))
    graph.add_node("embed_store", cast("Any", make_embed_store_node(db_engine, embedding_fn)))

    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "embed_store")
    graph.add_edge("embed_store", END)

    return graph.compile()
