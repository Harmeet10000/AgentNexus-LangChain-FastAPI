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
    embedding_fn: Any,
    redis: Any = None,
    graphiti: Any = None,
    reranker: CrossEncoderReranker | None = None,
) -> CompiledStateGraph:
    """Build a request-scoped retrieval graph over clauses."""
    query_llm = _structured(llm, QueryPlan)
    grader_llm = _structured(llm, ContextGrade)
    generator_llm = _structured(llm, GeneratedAnswer)

    graph = StateGraph(RetrievalState)
    graph.add_node("query_analyzer", cast("Any", make_query_analyzer_node(query_llm, redis)))
    graph.add_node("graph_neo4j", cast("Any", make_graph_retrieval_node(graphiti)))
    graph.add_node(
        "hybrid_postgres",
        cast("Any", make_hybrid_retrieval_node(repo, embedding_fn, redis)),
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
    return graph.compile()


def _structured(llm: Any, schema: type[Any]) -> Any:
    if hasattr(llm, "with_structured_output"):
        return llm.with_structured_output(schema)
    return llm
