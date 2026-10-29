"""Agent Saul LangGraph assembly entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Final

    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.graph.state import CompiledStateGraph

    from app.features.documents.application.tools.bundle import AgentToolBundle

    from .factory import SaulGraphNodes

from .factory import _build_graph_nodes, build_agent_registry
from .nodes import (
    _VALID_WORKER_NODES,
    dispatch_entity_extraction,
    make_state_hydration_node,
    route_after_qna,
    route_deep_research,
    route_from_orchestrator,
)
from .state import (
    GRAPH_NODE_NAMES,
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


def wire_saul_topology(graph: Any, node_for: Callable[[str], Any]) -> None:
    """The single source of Agent Saul's edge topology.

    ``node_for(name)`` supplies the node callable — the real nodes for the
    production graph, self-ending stubs for the sweep's read graph. Both
    builders MUST go through this function: a topology edited here without
    both graphs recompiling is a sweep that silently mis-reads checkpoints
    (missing node names make Pregel drop their pending tasks, so paused
    threads stop looking CLARIFYING).
    """
    for name in GRAPH_NODE_NAMES:
        graph.add_node(name, node_for(name))

    # Hydration runs before gateway on every run and is never a routing
    # target: it is wired explicitly so it stays out of GRAPH_NODE_NAMES,
    # _VALID_WORKER_NODES, and the orchestrator's dispatch table.
    graph.add_node("state_hydration", node_for("state_hydration"))
    graph.set_entry_point("state_hydration")
    graph.add_edge("state_hydration", "gateway")
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
            **{name: name for name in _VALID_WORKER_NODES},
            "planner_generate": "planner_generate",
            END: END,
        },
    )
    # Plan generation is deliberately a checkpoint boundary before approval.
    # The approval node is the only node containing interrupt(); resuming it
    # cannot re-run the planner LLM and replace the displayed proposal.
    graph.add_edge("planner_generate", "planner")
    graph.add_conditional_edges(
        "planner",
        route_deep_research,
        {
            "deep_research": "deep_research",
            "orchestrator": "orchestrator",
            "planner_generate": "planner_generate",
        },
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


def _wire_graph(graph: Any, nodes: SaulGraphNodes) -> None:
    wire_saul_topology(
        graph,
        lambda name: (
            make_state_hydration_node() if name == "state_hydration" else getattr(nodes, name)
        ),
    )


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
    # Input schema is the FULL state TypedDict — a state-compat input per
    # turn. A narrower Pydantic FILTERED input was verified to silently drop
    # keys outside it (probed: `messages` arrived as [], `status` absent),
    # losing the tagged turn query that §4.2's projector keys on and
    # crashing the qna router on a missing status. The WS layer's own DTOs
    # stay the request-validation boundary.
    graph: Any = state_graph_factory(
        LegalAgentState, input_schema=LegalAgentState, output_schema=LegalAgentOutputState
    )
    _wire_graph(graph=graph, nodes=nodes)

    return cast("CompiledStateGraph[Any]", graph.compile(checkpointer=checkpointer))
