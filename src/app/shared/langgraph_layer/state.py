"""
LangGraph state schemas.

All agents share the base AgentState.  Specialized agents extend it.
Uses Pydantic v2 + Annotated reducers for type-safe state mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypeVar

from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Base agent state (extends LangGraph's MessagesState)
# ---------------------------------------------------------------------------


class AgentState(MessagesState):
    """
    Core state shared by all agents.

    LangGraph's MessagesState provides:
      messages: Annotated[list[AnyMessage], add_messages]

    We extend it with production fields.
    """

    # Structured final output (set once the agent is done)
    structured_output: dict[str, Any] | None = None

    # Human-in-the-loop: pending action awaiting approval
    pending_approval: dict[str, Any] | None = None

    # Todo list managed by TodoListMiddleware
    todo_list: list[str] = []

    # Short-circuit flag: set True by guardrails to stop the graph
    blocked: bool = False
    block_reason: str | None = None

    # Retry counters
    model_retry_count: int = 0
    tool_retry_count: int = 0

    # Metadata propagated through the graph
    session_id: str | None = None
    user_id: str | None = None
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Multi-agent / supervisor state
# ---------------------------------------------------------------------------


class SupervisorState(AgentState):
    """State for the supervisor node in a multi-agent graph."""

    # Name of the next agent to call (set by router/supervisor)
    next_agent: str | None = None

    # Accumulated results from all sub-agents
    agent_results: Annotated[list[dict[str, Any]], lambda x, y: x + y] = []

    # Names of agents that have already run (loop prevention)
    completed_agents: list[str] = []

    # The final synthesized response
    final_response: str | None = None


# ---------------------------------------------------------------------------
# Context schemas (not persisted — passed at invocation time)
# ---------------------------------------------------------------------------


@dataclass
class BaseContext:
    """
    Minimal runtime context.  Passed via create_agent(context_schema=...).
    Values here are available in middleware and dynamic prompts but are NOT
    stored in LangGraph checkpoints.
    """

    user_id: str = ""
    user_role: str = "viewer"  # viewer | editor | admin
    session_id: str = ""


@dataclass
class RichContext(BaseContext):
    """Extended context with user preferences and feature flags."""

    user_name: str = ""
    language: str = "en"
    timezone: str = "UTC"
    feature_flags: dict[str, bool] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Typed config for LangGraph thread/runtime
# ---------------------------------------------------------------------------

StateT = TypeVar("StateT", bound=AgentState)
