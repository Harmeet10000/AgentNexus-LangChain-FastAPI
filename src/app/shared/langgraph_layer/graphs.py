"""
Graph builders.

Provides:
- build_single_agent_graph: ReAct loop with guardrails
- build_supervisor_graph: multi-agent with supervisor
- build_custom_workflow: deterministic pipeline

These are LOW-LEVEL graphs. For high-level agent creation use agents/factory.py.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph_layer.edges import after_guardrail, after_tools, should_continue
from langgraph_layer.nodes import (
    agent_node,
    guardrail_node,
    human_approval_node,
    synthesizer_node,
)
from langgraph_layer.state import AgentState, SupervisorState


def build_single_agent_graph(
    tools: list[Any],
    *,
    checkpointer: Any | None = None,
    enable_guardrails: bool = True,
    human_in_the_loop: bool = False,
) -> Any:
    """
    Build a production ReAct agent graph.

    Flow:
      START → agent → [tools → agent]* → guardrail → END
                              ↑                |
                         (loop back)           ↓
                                             END (if blocked)
    """
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    if enable_guardrails:
        graph.add_node("guardrail", guardrail_node)

    if human_in_the_loop:
        graph.add_node("human_approval", human_approval_node)

    # Edges
    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "guardrail": "guardrail" if enable_guardrails else END,
            "end": END,
        },
    )

    graph.add_edge("tools", "agent")

    if enable_guardrails:
        graph.add_conditional_edges(
            "guardrail",
            after_guardrail,
            {"end": END, "agent": "agent"},
        )

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"] if human_in_the_loop else [],
    )


def build_supervisor_graph(
    agent_nodes: dict[str, Any],  # {name: async_fn}
    *,
    checkpointer: Any | None = None,
) -> Any:
    """
    Build a supervisor-based multi-agent graph.

    Flow:
      START → supervisor → [agent_1 | agent_2 | ...] → supervisor → ... → synthesizer → END
    """
    from functools import partial

    from langgraph_layer.nodes import supervisor_node

    agent_names = list(agent_nodes.keys())

    graph = StateGraph(SupervisorState)

    # Supervisor node bound to agent names
    async def _supervisor(state: SupervisorState):
        return await supervisor_node(state, agent_names)

    graph.add_node("supervisor", _supervisor)
    graph.add_node("synthesizer", synthesizer_node)

    for name, fn in agent_nodes.items():
        graph.add_node(name, fn)
        # After each sub-agent completes, return to supervisor
        graph.add_edge(name, "supervisor")

    graph.add_edge(START, "supervisor")

    # Supervisor dynamically routes (Command-based, handled in node)
    # We add edges to all possible sub-agents
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s.get("next_agent", "synthesizer"),
        {**{name: name for name in agent_names}, "synthesizer": "synthesizer"},
    )

    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)


def build_sequential_workflow(
    steps: list[tuple[str, Any]],  # [(name, async_fn), ...]
    *,
    state_schema: type = AgentState,
    checkpointer: Any | None = None,
) -> Any:
    """
    Build a deterministic, sequential pipeline.
    Each step passes state to the next automatically.
    """
    graph = StateGraph(state_schema)

    for name, fn in steps:
        graph.add_node(name, fn)

    # Wire steps sequentially
    names = [n for n, _ in steps]
    graph.add_edge(START, names[0])
    for i in range(len(names) - 1):
        graph.add_edge(names[i], names[i + 1])
    graph.add_edge(names[-1], END)

    return graph.compile(checkpointer=checkpointer)


def build_parallel_fanout_graph(
    parallel_nodes: dict[str, Any],  # {name: async_fn}
    merge_node: Any,
    *,
    state_schema: type = AgentState,
    checkpointer: Any | None = None,
) -> Any:
    """
    Fan out to multiple nodes in parallel, then merge.

    Flow: START → [node_a | node_b | node_c] → merge → END
    """
    from langgraph.types import Send

    graph = StateGraph(state_schema)

    for name, fn in parallel_nodes.items():
        graph.add_node(name, fn)

    graph.add_node("merge", merge_node)

    # Fan-out: from START to all parallel nodes
    graph.add_edge(START, list(parallel_nodes.keys()))  # type: ignore[arg-type]

    for name in parallel_nodes:
        graph.add_edge(name, "merge")

    graph.add_edge("merge", END)

    return graph.compile(checkpointer=checkpointer)
