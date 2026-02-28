"""
LangGraph edge functions — determine the next node at runtime.

Edges are pure functions: (state) -> node_name | list[Send].
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph_layer.state import AgentState, SupervisorState

# ---------------------------------------------------------------------------
# Standard agent loop edges
# ---------------------------------------------------------------------------


def should_continue(
    state: AgentState,
) -> Literal["tools", "guardrail", "end"]:
    """
    After the agent node:
    - If the last AI message has tool_calls → go to tools
    - Otherwise → run guardrail then end
    """
    if state.get("blocked"):
        return "end"

    messages = state["messages"]
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)

    if last_ai and last_ai.tool_calls:
        return "tools"
    return "guardrail"


def after_guardrail(
    state: AgentState,
) -> Literal["end", "agent"]:
    """
    After the guardrail node:
    - If blocked → end (guardrail has already replaced the message)
    - Otherwise → end (normal completion)
    """
    if state.get("blocked"):
        return "end"
    return "end"


def after_tools(
    state: AgentState,
) -> Literal["agent"]:
    """Always return to agent after tool execution."""
    return "agent"


# ---------------------------------------------------------------------------
# Multi-agent routing edges
# ---------------------------------------------------------------------------


def supervisor_route(
    state: SupervisorState,
    agents: list[str],
) -> str:
    """
    Reads `next_agent` set by the supervisor node.
    Falls back to END if nothing is set.
    """
    from langgraph.graph import END

    next_agent = state.get("next_agent")
    if next_agent and next_agent in agents:
        return next_agent
    return END


def subagent_done(
    state: SupervisorState,
) -> Literal["supervisor"]:
    """After a sub-agent finishes, always return to the supervisor."""
    return "supervisor"


# ---------------------------------------------------------------------------
# Retry edges
# ---------------------------------------------------------------------------


def should_retry_model(
    state: AgentState,
    *,
    max_retries: int = 3,
) -> Literal["agent", "end"]:
    """Route back to the agent node if retry count is within limit."""
    if state.get("model_retry_count", 0) < max_retries:
        return "agent"
    return "end"


def should_retry_tool(
    state: AgentState,
    *,
    max_retries: int = 3,
) -> Literal["tools", "agent", "end"]:
    """Route back to tools if retry count within limit; else back to agent."""
    retries = state.get("tool_retry_count", 0)
    if retries < max_retries:
        return "tools"
    return "agent"
