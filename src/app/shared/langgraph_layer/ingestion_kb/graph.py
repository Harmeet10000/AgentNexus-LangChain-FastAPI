"""Contract KB ingestion graph factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .pipeline_node import (
    dispatch_contextualize_chunks,
    make_classify_extract_node,
    make_contextualize_chunk_node,
    make_embed_store_node,
    make_extract_schema_node,
    make_graphiti_upsert_node,
    make_parse_document_node,
    make_segment_document_node,
)
from .state import (
    ClauseSegmentationResult,
    ContextualizedChunk,
    ContractMetadata,
    EntityExtractionResult,
    IngestionState,
)

if TYPE_CHECKING:
    from typing import Any

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

    from .state import EmbeddingFunction


def build_ingestion_graph(
    extraction_llm: Any,
    db_engine: AsyncEngine,
    embedding_fn: EmbeddingFunction,
    graphiti_service: Any,
    redis: Redis | None = None,
) -> CompiledStateGraph:
    """Build the contract ingestion graph once during application startup."""
    graph = StateGraph(IngestionState)
    graph.add_node("parse_document", cast("Any", make_parse_document_node()))
    graph.add_node(
        "extract_schema",
        cast("Any", make_extract_schema_node(_structured(extraction_llm, ContractMetadata))),
    )
    graph.add_node(
        "segment_document",
        cast(
            "Any", make_segment_document_node(_structured(extraction_llm, ClauseSegmentationResult))
        ),
    )
    graph.add_node(
        "contextualize_chunks",
        cast(
            "Any", make_contextualize_chunk_node(_structured(extraction_llm, ContextualizedChunk))
        ),
    )
    graph.add_node(
        "classify_extract_entities",
        cast(
            "Any", make_classify_extract_node(_structured(extraction_llm, EntityExtractionResult))
        ),
    )
    graph.add_node(
        "embed_store",
        cast("Any", make_embed_store_node(db_engine, embedding_fn, redis)),
    )
    graph.add_node("graphiti_upsert", cast("Any", make_graphiti_upsert_node(graphiti_service)))

    graph.set_entry_point("parse_document")
    graph.add_edge("parse_document", "extract_schema")
    graph.add_edge("extract_schema", "segment_document")
    graph.add_conditional_edges("segment_document", dispatch_contextualize_chunks)
    graph.add_edge("contextualize_chunks", "classify_extract_entities")
    graph.add_edge("classify_extract_entities", "embed_store")
    graph.add_edge("embed_store", "graphiti_upsert")
    graph.add_edge("graphiti_upsert", END)

    return graph.compile()


def _structured(llm: Any, schema: type[Any]) -> Any:
    if hasattr(llm, "with_structured_output"):
        return llm.with_structured_output(schema)
    return llm
