"""Retrieval graph factory for canonical clauses KB."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    make_context_grader_node,
    make_generator_node,
    make_graph_retrieval_node,
    make_hybrid_retrieval_node,
    make_query_analyzer_node,
    make_reranker_node,
    should_retry_or_generate,
    should_run_graph,
)
from .state import ContextGrade, GeneratedAnswer, QueryPlan, RetrievalState

if TYPE_CHECKING:
    from typing import Any

    from .reranker import CrossEncoderReranker


def build_retrieval_graph(
    *,
    llm: Any,
    repo: Any,
    redis: Any = None,
    graphiti: Any = None,
    reranker: CrossEncoderReranker | None = None,
) -> CompiledStateGraph[Any]:
    """Build a request-scoped retrieval graph over clauses.

    There is deliberately no ``embedding_fn`` parameter. It was annotated ``Any``, which erased
    the contract and forced the node to duck-type the injected object through three candidate
    method names — two of which no caller ever exercised. The embedding path is now resolved
    where it is used, so the client is process-wide and its task type is declared per call.
    """
    query_llm = _structured(llm, QueryPlan)
    grader_llm = _structured(llm, ContextGrade)
    generator_llm = _structured(llm, GeneratedAnswer)

    graph = StateGraph(RetrievalState)  # ty: ignore[invalid-argument-type]
    graph.add_node("query_analyzer", cast("Any", make_query_analyzer_node(query_llm, redis)))
    graph.add_node("graph_neo4j", cast("Any", make_graph_retrieval_node(graphiti)))
    graph.add_node(
        "hybrid_postgres",
        cast("Any", make_hybrid_retrieval_node(repo, redis)),
    )
    graph.add_node("reranker", cast("Any", make_reranker_node(reranker)))
    graph.add_node("context_grader", cast("Any", make_context_grader_node(grader_llm)))
    graph.add_node("generate", cast("Any", make_generator_node(generator_llm, redis)))

    graph.set_entry_point("query_analyzer")
    graph.add_conditional_edges(
        "query_analyzer",
        should_run_graph,
        {"graph": "graph_neo4j", "hybrid": "hybrid_postgres", "generate": "generate"},
    )
    graph.add_edge("graph_neo4j", "hybrid_postgres")
    graph.add_edge("hybrid_postgres", "reranker")
    graph.add_edge("reranker", "context_grader")
    graph.add_conditional_edges(
        "context_grader",
        should_retry_or_generate,
        {"query_analyzer": "query_analyzer", "generate": "generate"},
    )
    graph.add_edge("generate", END)
    return graph.compile()  # ty: ignore[invalid-return-type]


def _structured(llm: Any, schema: type[Any]) -> Any:
    if hasattr(llm, "with_structured_output"):
        return llm.with_structured_output(schema)
    return llm
