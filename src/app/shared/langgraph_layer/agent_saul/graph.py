"""Agent Saul LangGraph assembly entrypoint."""

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .factory import SaulGraphNodes, _build_graph_nodes, build_agent_registry
from .nodes import dispatch_entity_extraction, route_after_qna, route_from_orchestrator
from .state import LegalAgentState


def _wire_graph(graph: Any, nodes: SaulGraphNodes) -> None:
    graph.add_node("gateway", nodes.gateway)
    graph.add_node("qna", nodes.qna)
    graph.add_node("orchestrator", nodes.orchestrator)
    graph.add_node("planner", nodes.planner)
    graph.add_node("ingestion", nodes.ingestion)
    graph.add_node("normalization", nodes.normalization)
    graph.add_node("segmentation", nodes.segmentation)
    graph.add_node("entity_extraction", nodes.entity_extraction)
    graph.add_node("relationship_mapping", nodes.relationship_mapping)
    graph.add_node("risk_analysis", nodes.risk_analysis)
    graph.add_node("compliance", nodes.compliance)
    graph.add_node("grounding_verification", nodes.grounding_verification)
    graph.add_node("human_review", nodes.human_review)
    graph.add_node("finalization", nodes.finalization)
    graph.add_node("persist_memory", nodes.persist_memory)

    graph.set_entry_point("gateway")
    graph.add_edge("gateway", "qna")
    graph.add_conditional_edges(
        "qna",
        route_after_qna,
        {"qna": "qna", "orchestrator": "orchestrator"},
    )
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "planner": "planner",
            "ingestion": "ingestion",
            "finalization": "finalization",
            END: END,
        },
    )
    graph.add_edge("planner", "orchestrator")
    graph.add_edge("ingestion", "normalization")
    graph.add_edge("normalization", "segmentation")
    graph.add_conditional_edges("segmentation", dispatch_entity_extraction)
    graph.add_edge("entity_extraction", "relationship_mapping")
    graph.add_edge("relationship_mapping", "risk_analysis")
    graph.add_edge("relationship_mapping", "compliance")
    graph.add_edge("risk_analysis", "grounding_verification")
    graph.add_edge("compliance", "grounding_verification")
    graph.add_edge("grounding_verification", "human_review")
    graph.add_edge("human_review", "orchestrator")
    graph.add_edge("finalization", "persist_memory")
    graph.add_edge("persist_memory", END)


def build_saul_graph(
    checkpointer: AsyncPostgresSaver,
    pro_llm: BaseChatModel,
    flash_llm: BaseChatModel,
    cognee_client: Any,
) -> CompiledStateGraph:
    """Build and compile the Agent Saul LangGraph."""
    registry = build_agent_registry(pro_llm, flash_llm)
    nodes = _build_graph_nodes(registry=registry, pro_llm=pro_llm, cognee_client=cognee_client)

    state_graph_factory = cast("Any", StateGraph)
    graph: Any = state_graph_factory(LegalAgentState)
    _wire_graph(graph=graph, nodes=nodes)

    return cast("CompiledStateGraph", graph.compile(checkpointer=checkpointer))
