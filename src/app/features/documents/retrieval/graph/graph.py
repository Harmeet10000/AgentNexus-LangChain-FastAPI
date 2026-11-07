"""Retrieval graph factory for the canonical unified-chunk KB."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    make_context_grader_node,
    make_edge_grounding_node,
    make_generator_node,
    make_graph_retrieval_node,
    make_hybrid_retrieval_node,
    make_post_process_node,
    make_query_analyzer_node,
    make_reranker_node,
    make_source_identifier_node,
    should_retry_or_generate,
    should_run_graph,
)
from .state import RetrievalGraphContext, RetrievalState

if TYPE_CHECKING:
    from typing import Any


def build_retrieval_graph() -> CompiledStateGraph[Any]:
    """Build the process-scoped retrieval graph over unified chunks.

    There is deliberately no ``embedding_fn`` parameter. It was annotated ``Any``, which erased
    the contract and forced the node to duck-type the injected object through three candidate
    method names — two of which no caller ever exercised. The embedding path is now resolved
    where it is used, so the client is process-wide and its task type is declared per call.

    Compile-once provisioning (`compiled-graph-lifecycle`, mount-retrieval-graph task 3.1):
    this factory takes **no** collaborators. Every request-scoped collaborator — the language
    model, repository, cache client, relationship-graph client, reranker, and the optional
    grader seam — arrives per invocation through ``ainvoke(..., context=RetrievalGraphContext
    ...)``. Compiling once is what makes the graph a first-class provisioned surface; nothing
    request-scoped is captured in any node closure.

    The graph carries a context schema (`RetrievalGraphContext`) so per-invocation
    collaborators arrive through ``ainvoke(..., context=...)`` instead of being captured at
    compile time (`compiled-graph-lifecycle`).
    """
    graph = StateGraph(RetrievalState, context_schema=RetrievalGraphContext)  # ty: ignore[invalid-argument-type]
    graph.add_node("query_analyzer", cast("Any", make_query_analyzer_node()))
    graph.add_node("source_identifier", cast("Any", make_source_identifier_node()))
    graph.add_node("graph_neo4j", cast("Any", make_graph_retrieval_node()))
    graph.add_node("hybrid_postgres", cast("Any", make_hybrid_retrieval_node()))
    graph.add_node("reranker", cast("Any", make_reranker_node()))
    graph.add_node("post_process", cast("Any", make_post_process_node()))
    graph.add_node("edge_grounding", cast("Any", make_edge_grounding_node()))
    graph.add_node("context_grader", cast("Any", make_context_grader_node()))
    graph.add_node("generate", cast("Any", make_generator_node()))

    graph.set_entry_point("query_analyzer")
    graph.add_conditional_edges(
        "query_analyzer",
        should_run_graph,
        {
            "graph": "source_identifier",
            "hybrid": "source_identifier",
            "generate": "generate",
        },
    )
    graph.add_conditional_edges(
        "source_identifier",
        # Only the retrieval routes can reach this node (the analyzer sends
        # cache hits and trivial queries straight to `generate`), so only
        # they are mapped here. A `generate` arm would be an edge that fires
        # never — topology as documentation must stay honest.
        should_run_graph,
        {"graph": "graph_neo4j", "hybrid": "hybrid_postgres"},
    )
    graph.add_edge("graph_neo4j", "hybrid_postgres")
    graph.add_edge("hybrid_postgres", "reranker")
    graph.add_edge("reranker", "post_process")
    graph.add_edge("post_process", "edge_grounding")
    graph.add_edge("edge_grounding", "context_grader")
    graph.add_conditional_edges(
        "context_grader",
        should_retry_or_generate,
        {"query_analyzer": "query_analyzer", "generate": "generate"},
    )
    graph.add_edge("generate", END)
    return graph.compile()  # ty: ignore[invalid-return-type]
