"""
Subagent tool — wraps any compiled LangGraph agent as a callable LangChain tool.

This enables nested agent calls: a parent agent can delegate to specialized
sub-agents via a tool call, receiving structured results back.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubagentInput(BaseModel):
    task: str = Field(..., description="The task or question to delegate to this sub-agent.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context to pass to the sub-agent.",
    )


class SubagentOutput(BaseModel):
    result: str
    success: bool
    agent_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def make_subagent_tool(
    name: str,
    description: str,
    agent: Any,  # Compiled LangGraph agent (CompiledStateGraph)
    *,
    thread_id: str | None = None,
    context_schema: type | None = None,
    result_key: str = "messages",
) -> StructuredTool:
    """
    Wrap a compiled LangGraph agent as a StructuredTool.

    Args:
        name: Tool name (shown to the LLM calling the tool).
        description: Tool description with usage guidance.
        agent: A compiled LangGraph CompiledStateGraph.
        thread_id: Optional fixed thread ID for stateful sub-agent.
        context_schema: Context dataclass to pass at invocation time.
        result_key: State key to read the result from.
    """
    agent_name = name

    async def _invoke(task: str, context: dict[str, Any] | None = None) -> str:
        config: dict[str, Any] = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}
        if context and context_schema:
            config["context"] = context_schema(**context)

        try:
            state = await agent.ainvoke(
                {"messages": [("user", task)]},
                config=config,
            )
            # Extract last AI message or structured output
            if "structured_output" in state and state["structured_output"]:
                result = json.dumps(state["structured_output"])
            else:
                messages = state.get("messages", [])
                ai_messages = [m for m in messages if hasattr(m, "content") and not hasattr(m, "tool_calls")]
                result = ai_messages[-1].content if ai_messages else "No response"
                if not isinstance(result, str):
                    result = str(result)

            output = SubagentOutput(
                result=result,
                success=True,
                agent_name=agent_name,
            )
            return output.model_dump_json()

        except Exception as exc:
            logger.exception("Subagent %s failed: %s", agent_name, exc)
            output = SubagentOutput(
                result=f"Sub-agent failed: {exc}",
                success=False,
                agent_name=agent_name,
            )
            return output.model_dump_json()

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
        args_schema=SubagentInput,
        return_direct=False,
    )
