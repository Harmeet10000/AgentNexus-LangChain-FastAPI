"""
Agent factory — the main entry point for creating production agents.

Uses LangChain 1.0's `create_agent` with:
- context_schema for typed runtime context
- middleware stack for context engineering
- response_format for structured outputs
- checkpointer for persistence
- store for long-term memory

NOTE on naming:
  This project's `langchain/` and `langgraph/` folders will conflict with
  the installed packages of the same name if they are at the Python import
  root.  Place them inside your app package (e.g., `src/myapp/langchain/`)
  so that relative imports work, or use the `langchain_layer/` /
  `langgraph_layer/` naming shown here.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agents.memory.manager import MemoryManager
from agents.tools.base import registry as tool_registry
from config.settings import get_settings
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_layer.models import build_chat_model, build_fast_model
from langchain_layer.prompts import AGENT_SYSTEM_PROMPT, SystemPromptParts
from langgraph_layer.state import BaseContext, RichContext

logger = logging.getLogger(__name__)
_settings = get_settings()


# ---------------------------------------------------------------------------
# Agent spec
# ---------------------------------------------------------------------------


@dataclass
class AgentSpec:
    """
    Declarative specification for a production agent.
    Pass to `create_production_agent`.
    """

    # Identity
    name: str
    description: str = ""

    # Model
    model_name: str | None = None  # defaults to settings
    temperature: float = 0.0
    max_tokens: int | None = None

    # Tools (names from registry, or BaseTool instances)
    tools: list[str | BaseTool] = field(default_factory=list)

    # Prompt
    system_prompt: str | SystemPromptParts | None = None

    # Context schema (dataclass)
    context_schema: type = BaseContext

    # Structured output schema (Pydantic model), or None for text
    response_format: type | None = None

    # Memory
    memory_backend: str = "memory"  # memory | postgres | redis
    enable_long_term_memory: bool = True

    # Middleware config
    enable_guardrails: bool = True
    enable_tool_selector: bool = True
    enable_human_loop: bool = False
    human_loop_tools: dict[str, bool] | None = None
    max_tokens_before_summary: int = 4000
    messages_to_keep: int = 8

    # Additional middleware to inject
    extra_middleware: list[Any] = field(default_factory=list)

    # LangGraph options
    debug: bool = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_production_agent(spec: AgentSpec) -> "ProductionAgent":
    """
    Build a fully configured production agent from a spec.

    Returns a ProductionAgent with async invoke, stream, and batch methods.
    """
    from middleware import build_default_middleware_stack

    # Resolve tools
    resolved_tools: list[BaseTool] = []
    for t in spec.tools:
        if isinstance(t, str):
            resolved_tools.append(tool_registry.get(t))
        else:
            resolved_tools.append(t)

    # Build middleware stack
    middleware = build_default_middleware_stack(
        max_tokens_before_summary=spec.max_tokens_before_summary,
        messages_to_keep=spec.messages_to_keep,
        enable_guardrails=spec.enable_guardrails,
        enable_tool_selector=spec.enable_tool_selector and bool(resolved_tools),
        enable_human_loop=spec.enable_human_loop,
        human_loop_tools=spec.human_loop_tools,
    ) + spec.extra_middleware

    # Build memory
    memory = MemoryManager(backend=spec.memory_backend)

    # System prompt
    if spec.system_prompt is None:
        system_text = AGENT_SYSTEM_PROMPT.build()
    elif isinstance(spec.system_prompt, SystemPromptParts):
        system_text = spec.system_prompt.build()
    else:
        system_text = spec.system_prompt

    # Build LangChain model
    model = build_chat_model(
        model_name=spec.model_name,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
    )

    # create_agent (LangChain 1.0)
    compiled = create_agent(
        model,
        tools=resolved_tools,
        system_prompt=system_text,
        middleware=middleware,
        response_format=spec.response_format,
        context_schema=spec.context_schema,
        checkpointer=memory.checkpointer,
        debug=spec.debug,
        name=spec.name,
    )

    return ProductionAgent(
        spec=spec,
        compiled=compiled,
        memory=memory,
    )


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------


@dataclass
class ProductionAgent:
    """
    Wraps a compiled LangGraph agent with production runtime behaviour:
    - Long-term memory injection
    - Session saving
    - Async invoke, stream, and batch
    """

    spec: AgentSpec
    compiled: Any  # CompiledStateGraph
    memory: MemoryManager

    async def ainvoke(
        self,
        user_message: str,
        *,
        thread_id: str,
        context: Any | None = None,
        user_id: str = "default",
        save_memory: bool = True,
    ) -> dict[str, Any]:
        """
        Single async invocation.

        Args:
            user_message: The user's input message.
            thread_id: LangGraph thread ID for conversation persistence.
            context: Instance of spec.context_schema.
            user_id: User ID for long-term memory.
            save_memory: Whether to save session to long-term memory afterwards.
        """
        config = self._build_config(thread_id=thread_id, context=context)
        input_state = {"messages": [("user", user_message)]}

        # Inject long-term memory into the first invocation
        if self.spec.enable_long_term_memory:
            from langchain_core.messages import HumanMessage
            msgs = await self.memory.inject_long_term_context(
                [HumanMessage(content=user_message)],
                user_id=user_id,
                agent_id=self.spec.name,
            )
            input_state = {"messages": msgs}

        result = await self.compiled.ainvoke(input_state, config=config)

        if save_memory and self.spec.enable_long_term_memory:
            await self.memory.save_session(
                result.get("messages", []),
                user_id=user_id,
                session_id=thread_id,
                agent_id=self.spec.name,
            )

        return result

    async def astream(
        self,
        user_message: str,
        *,
        thread_id: str,
        context: Any | None = None,
        user_id: str = "default",
        stream_mode: str = "messages",
    ) -> AsyncIterator[Any]:
        """
        Stream the agent's response token by token.

        stream_mode options: "messages" (tokens) | "updates" (node-level) | "values"

        Usage in FastAPI::

            @router.get("/stream")
            async def stream_endpoint():
                async def event_generator():
                    async for chunk in agent.astream("Hello", thread_id="t1"):
                        yield f"data: {chunk}\\n\\n"
                return StreamingResponse(event_generator(), media_type="text/event-stream")
        """
        config = self._build_config(thread_id=thread_id, context=context)
        input_state = {"messages": [("user", user_message)]}

        async for chunk in self.compiled.astream(
            input_state,
            config=config,
            stream_mode=stream_mode,
        ):
            yield chunk

    async def abatch(
        self,
        messages: list[str],
        *,
        thread_ids: list[str],
        context: Any | None = None,
        user_id: str = "default",
        max_concurrency: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Batch invoke the agent concurrently on multiple messages.
        Each message gets its own thread_id for isolated history.
        """
        import asyncio

        if len(messages) != len(thread_ids):
            raise ValueError("messages and thread_ids must have the same length")

        max_c = max_concurrency or _settings.model.max_concurrency
        semaphore = asyncio.Semaphore(max_c)

        async def bounded_invoke(msg: str, tid: str) -> dict[str, Any]:
            async with semaphore:
                return await self.ainvoke(msg, thread_id=tid, context=context, user_id=user_id)

        return await asyncio.gather(
            *[bounded_invoke(m, t) for m, t in zip(messages, thread_ids)]
        )

    async def resume_after_approval(
        self,
        thread_id: str,
        *,
        context: Any | None = None,
    ) -> dict[str, Any]:
        """
        Resume a paused (human-in-the-loop) agent after human approval.
        Call this after updating the agent's state via LangGraph's update_state.
        """
        config = self._build_config(thread_id=thread_id, context=context)
        return await self.compiled.ainvoke(None, config=config)

    def _build_config(self, *, thread_id: str, context: Any | None = None) -> dict[str, Any]:
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if context is not None:
            config["context"] = context
        return config

    def get_state(self, thread_id: str) -> Any:
        """Get current checkpoint state for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.compiled.get_state(config)

    def get_state_history(self, thread_id: str) -> Any:
        """Get full checkpoint history for time-travel debugging."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.compiled.get_state_history(config)
