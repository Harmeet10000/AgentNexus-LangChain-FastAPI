"""Agent Saul LangGraph assembly entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from typing import Any, Final

    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph.state import CompiledStateGraph

    from app.shared.rag.graphiti.registry import AgentToolBundle

    from .factory import SaulGraphNodes

from .factory import _build_graph_nodes, build_agent_registry
from .nodes import (
    _VALID_WORKER_NODES,
    dispatch_entity_extraction,
    route_after_qna,
    route_deep_research,
    route_from_orchestrator,
)
from .state import (
    GRAPH_NODE_NAMES,
    LegalAgentInputState,
    LegalAgentOutputState,
    LegalAgentState,
)

# Step budget (agent-state-handoff): the maximum number of super-steps a single
# run may take before LangGraph aborts it with GraphRecursionError. Pinned
# explicitly rather than inherited from the library default so a default
# change surfaces as a diff here. Invocation sites pass it via
# `saul_runtime_config`; `compile()` takes no recursion limit, so the budget
# lives on the invoke config, not on the compiled graph.
SAUL_STEP_BUDGET: Final[int] = 25


def saul_runtime_config(thread_id: str) -> dict[str, Any]:
    """Invoke config for Agent Saul runs: thread scoping plus the step budget."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": SAUL_STEP_BUDGET,
    }


def _wire_graph(graph: Any, nodes: SaulGraphNodes) -> None:
    for name in GRAPH_NODE_NAMES:
        graph.add_node(name, getattr(nodes, name))

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
        {name: name for name in _VALID_WORKER_NODES} | {"planner": "planner", END: END},
    )
    graph.add_conditional_edges(
        "planner",
        route_deep_research,
        {"deep_research": "deep_research", "orchestrator": "orchestrator"},
    )
    graph.add_edge("deep_research", "orchestrator")
    graph.add_edge("ingestion", "normalization")
    graph.add_edge("normalization", "segmentation")
    graph.add_conditional_edges("segmentation", dispatch_entity_extraction)
    graph.add_edge("entity_extraction", "relationship_mapping")
    graph.add_edge("relationship_mapping", "risk_analysis")
    graph.add_edge("risk_analysis", "compliance")
    graph.add_edge("compliance", "grounding_verification")
    graph.add_edge("grounding_verification", "human_review")
    graph.add_edge("human_review", "orchestrator")
    graph.add_edge("finalization", "persist_memory")
    graph.add_edge("persist_memory", END)


def build_saul_graph(
    checkpointer: AsyncPostgresSaver,
    pro_llm: BaseChatModel,
    flash_llm: BaseChatModel,
    memory_service: Any,
    tool_registry: AgentToolBundle,
) -> CompiledStateGraph[Any]:
    """Build and compile the Agent Saul LangGraph."""
    registry = build_agent_registry(pro_llm, flash_llm, tools=tool_registry)
    nodes = _build_graph_nodes(
        registry=registry,
        pro_llm=pro_llm,
        memory_service=memory_service,
        tool_registry=tool_registry,
    )

    state_graph_factory = cast("Any", StateGraph)
    graph: Any = state_graph_factory(
        LegalAgentState, input_schema=LegalAgentInputState, output_schema=LegalAgentOutputState
    )
    _wire_graph(graph=graph, nodes=nodes)

    return cast("CompiledStateGraph[Any]", graph.compile(checkpointer=checkpointer))
