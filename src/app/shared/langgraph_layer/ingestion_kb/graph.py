"""Contract KB ingestion graph factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.utils import logger

from .nodes import (
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
    from typing import Any, Final

    from langgraph.checkpoint.base import BaseCheckpointSaver
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine

#: Durability mode for ingestion checkpoints (Decision 8): checkpoints persist
#: while the next stage executes rather than only at completion, so a crash
#: mid-execution resumes from the last completed stage boundary instead of
#: replaying the whole document. Passed at invocation
#: (`graph.ainvoke(..., durability=INGESTION_DURABILITY)`), not at compile
#: time — the installed builder takes the checkpointer at compile time and the
#: durability mode per call.
INGESTION_DURABILITY: Final[str] = "async"


def ingestion_thread_config(*, user_id: str, content_hash: str) -> dict[str, object]:
    """Build the invocation config carrying the checkpoint thread identity.

    Thread identity derives from the document identity — the same pair D15
    names (`user_id`, `content_hash`) — so resubmitting a document resumes its
    thread instead of starting a second one. It travels as invocation config,
    never as pipeline state.
    """
    return {"configurable": {"thread_id": f"ingestion:{user_id}:{content_hash}"}}


def build_ingestion_graph(
    extraction_llm: Any,
    db_engine: AsyncEngine,
    graphiti_service: Any,
    redis: Redis | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any]:
    """Build the contract ingestion graph once during application startup.

    There is deliberately no ``embedding_fn`` parameter — see ``build_retrieval_graph`` for why
    injecting one was what allowed the duck-typing and the missing task type to coexist.

    The checkpointer is owned by the constructing process (the queue worker —
    D17 keeps the application-lifespan construction commented, so the app never
    passes one). Built without it, the graph runs without persistence and that
    is recorded here rather than implied.
    """
    if checkpointer is None:
        logger.bind(operation="build_ingestion_graph").warning(
            "ingestion_graph_built_without_checkpointer"
        )
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
        cast("Any", make_embed_store_node(db_engine, redis)),
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

    return graph.compile(checkpointer=checkpointer)


def _structured(llm: Any, schema: type[Any]) -> Any:
    if hasattr(llm, "with_structured_output"):
        return llm.with_structured_output(schema)
    return llm
