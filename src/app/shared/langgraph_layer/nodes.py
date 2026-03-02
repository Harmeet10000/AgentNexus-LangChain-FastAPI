"""
LangGraph node definitions.

Each node is a plain async function: (state) -> partial_state_update.
Nodes are composed in graph.py.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_layer.models import build_chat_model
from langchain_layer.prompts import AGENT_SYSTEM_PROMPT
from langgraph.types import Command
from langgraph_layer.state import AgentState, SupervisorState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent node (LLM call)
# ---------------------------------------------------------------------------


async def agent_node(state: AgentState) -> dict[str, Any]:
    """
    Core agent node: invokes the LLM with the current message history.
    Returns the AI message to be appended to state.messages.
    """
    llm = build_chat_model()
    # System prompt is the first message when not using create_agent
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT.build())] + list(messages)

    response = await llm.ainvoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Tool node (handled by LangGraph's ToolNode — see graph.py)
# Defining custom logic here for structured tool results.
# ---------------------------------------------------------------------------


async def format_tool_result_node(state: AgentState) -> dict[str, Any]:
    """
    Post-processes raw tool results into structured form before the next LLM call.
    Runs between the ToolNode and the agent node.
    """
    # In most cases, ToolNode handles this automatically.
    # This node exists as an extension point for custom formatting.
    return {}


# ---------------------------------------------------------------------------
# Guardrail node
# ---------------------------------------------------------------------------


async def guardrail_node(state: AgentState) -> dict[str, Any]:
    """
    Runs deterministic + model-based guardrails on the last AI message.
    Sets state.blocked = True if the response is unsafe.
    """
    from middleware.guardrails import evaluate_response

    messages = state["messages"]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    if not ai_messages:
        return {}

    last_ai = ai_messages[-1]
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    last_user = user_messages[-1] if user_messages else None

    result = await evaluate_response(
        user_input=last_user.content if last_user else "",
        ai_output=last_ai.content
        if isinstance(last_ai.content, str)
        else str(last_ai.content),
    )

    if not result["safe"]:
        logger.warning("Guardrail triggered: %s", result["reason"])
        return {
            "blocked": True,
            "block_reason": result["reason"],
            "messages": [
                AIMessage(
                    content=(
                        "I'm unable to fulfill that request. "
                        f"Reason: {result['reason']}"
                    )
                )
            ],
        }
    return {}


# ---------------------------------------------------------------------------
# Supervisor node (multi-agent)
# ---------------------------------------------------------------------------


async def supervisor_node(state: SupervisorState, agents: list[str]) -> Command:
    """
    Supervisor that routes to the next agent or finishes.
    Returns a LangGraph Command to dynamically select the next node.
    """
    from langchain_layer.chains import build_router_chain

    completed = state.get("completed_agents", [])
    remaining = [a for a in agents if a not in completed]

    if not remaining:
        return Command(goto="synthesizer")

    # Let the LLM decide which agent goes next
    router = build_router_chain(remaining)
    messages = state["messages"]
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    query = last_human.content if last_human else ""

    result = await router.ainvoke({"input": query, "history": []})
    next_agent = result.get("agent", remaining[0])

    if next_agent not in remaining:
        next_agent = remaining[0]

    logger.info("Supervisor routing to: %s", next_agent)
    return Command(
        update={"next_agent": next_agent},
        goto=next_agent,
    )


# ---------------------------------------------------------------------------
# Synthesizer node (multi-agent — merges sub-agent outputs)
# ---------------------------------------------------------------------------


async def synthesizer_node(state: SupervisorState) -> dict[str, Any]:
    """
    Combines results from all sub-agents into a final coherent response.
    """
    llm = build_chat_model()
    agent_results = state.get("agent_results", [])
    if not agent_results:
        return {}

    results_text = "\n\n".join(
        f"### {r.get('agent', 'Agent')}\n{r.get('output', '')}" for r in agent_results
    )

    response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are a synthesis agent. Combine the following sub-agent outputs "
                    "into a single, coherent, well-structured response for the user."
                )
            ),
            HumanMessage(content=results_text),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop resume node
# ---------------------------------------------------------------------------


async def human_approval_node(state: AgentState) -> dict[str, Any]:
    """
    Pauses the graph here — LangGraph will interrupt_before this node.
    After human review, invoke the graph again with updated state.
    """
    # This node intentionally does nothing; it's the interrupt point.
    # The agent resumes when `.invoke()` is called again post-approval.
    return {}
