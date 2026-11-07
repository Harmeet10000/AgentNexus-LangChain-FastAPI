"""Pin the retrieval graph's node/edge shape (agentic-retrieval task 1.2).

A LangGraph topology change is a handful of `add_node`/`add_edge` calls — easy
to review past and invisible in a test summary. This test makes "this change
adds exactly N nodes and M edges" an assertion: any topology edit must update
the pinned sets below in the same commit.
"""

from __future__ import annotations

from app.features.documents.retrieval.graph.graph import build_retrieval_graph


def _shape() -> tuple[set[str], set[tuple[str, str]]]:
    # Compile-once (mount-retrieval-graph 3.1): the factory takes no
    # collaborators; everything arrives per invocation via the context.
    graph = build_retrieval_graph()
    drawable = graph.get_graph()
    return (
        set(drawable.nodes.keys()),
        {(edge.source, edge.target) for edge in drawable.edges},
    )


def test_retrieval_graph_node_set_is_pinned() -> None:
    nodes, _ = _shape()
    assert nodes == {
        "__start__",
        "__end__",
        "query_analyzer",
        "source_identifier",
        "graph_neo4j",
        "hybrid_postgres",
        "reranker",
        "post_process",
        "edge_grounding",
        "context_grader",
        "generate",
    }


def test_retrieval_graph_edge_set_is_pinned() -> None:
    # Net delta of the edge-grounding insertion (mount-retrieval-graph 1.1):
    # +1 node; the post_process → context_grader edge now routes through the
    # fail-closed edge_grounding node (post_process → edge_grounding →
    # context_grader).
    _, edges = _shape()
    assert edges == {
        ("__start__", "query_analyzer"),
        ("query_analyzer", "source_identifier"),
        ("query_analyzer", "generate"),
        ("source_identifier", "graph_neo4j"),
        ("source_identifier", "hybrid_postgres"),
        ("graph_neo4j", "hybrid_postgres"),
        ("hybrid_postgres", "reranker"),
        ("reranker", "post_process"),
        ("post_process", "edge_grounding"),
        ("edge_grounding", "context_grader"),
        ("context_grader", "query_analyzer"),
        ("context_grader", "generate"),
        ("generate", "__end__"),
    }
