"""
Multi-agent orchestration patterns.

Supports:
1. Supervisor  — LLM-driven routing to specialised sub-agents
2. Handoff     — Agent explicitly passes control to another agent
3. Router      — Deterministic or LLM-based routing at graph level
4. Skills      — Lightweight function-based capabilities (no full agent needed)
5. Custom workflow — Deterministic pipeline of agents
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agents.factory import AgentSpec, ProductionAgent, create_production_agent
from agents.tools.subagent import make_subagent_tool
from agents.tools.base import build_validation_error_handler
from langchain_core.tools import StructuredTool, tool
from langchain_layer.chains import build_router_chain
from langgraph.types import Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill — lightweight callable (no full agent overhead)
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """
    A named, async callable that represents a focused capability.
    Skills are cheaper than full agents and can be invoked directly
    by a supervisor or router.
    """

    name: str
    description: str
    fn: Callable[..., Any]
    tags: list[str] = field(default_factory=list)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------


@dataclass
class Handoff:
    """
    Represents a transfer of control from one agent to another.
    Include in an agent's tool list to enable explicit handoffs.
    """

    target_agent: str
    reason: str = ""

    def to_command(self) -> Command:
        return Command(goto=self.target_agent, update={"next_agent": self.target_agent})


class HandoffToolInput(BaseModel):
    """Validated input for explicit supervisor handoff tools."""

    reason: str = Field(default="", description="Why control should be transferred.")


def make_handoff_tool(target_agent: str, description: str) -> Any:
    """
    Create a LangChain tool that triggers a handoff to `target_agent`.
    Include this in an agent's tools so it can explicitly delegate.

    Usage::

        tools = [
            make_handoff_tool("research_agent", "Delegate research tasks"),
            make_handoff_tool("code_agent", "Delegate coding tasks"),
        ]
    """
    @tool(
        name=f"handoff_to_{target_agent}",
        args_schema=HandoffToolInput,
        handle_tool_error=True,
        handle_validation_error=build_validation_error_handler(HandoffToolInput),
        return_direct=True,
    )
    async def _handoff(reason: str = "") -> Command:
        """Transfer control to a specialised agent."""
        logger.info("Handoff to %s: %s", target_agent, reason)
        return Command(goto=target_agent, update={"next_agent": target_agent})

    _handoff.__doc__ = description
    return _handoff


# ---------------------------------------------------------------------------
# Supervisor-based multi-agent system
# ---------------------------------------------------------------------------


@dataclass
class MultiAgentSystem:
    """
    Supervisor-coordinated multi-agent system.

    The supervisor LLM decides which agent runs next.
    After all agents have responded, a synthesizer merges the results.
    """

    name: str = "multi_agent_system"
    supervisor_prompt: str = (
        "You are a supervisor managing a team of specialized agents. "
        "Analyze the user request and delegate to the most appropriate agent. "
        "After each agent responds, decide if more agents are needed or if you're done."
    )
    memory_backend: str = "memory"

    _agents: dict[str, ProductionAgent] = field(default_factory=dict)
    _skills: dict[str, Skill] = field(default_factory=dict)
    _compiled: Any = field(default=None, init=False)

    def register_agent(self, agent: ProductionAgent) -> MultiAgentSystem:
        self._agents[agent.spec.name] = agent
        return self

    def register_skill(self, skill: Skill) -> MultiAgentSystem:
        self._skills[skill.name] = skill
        return self

    def build(self) -> Any:
        """Compile the supervisor graph."""
        # Build agent tools (each sub-agent becomes a tool for the supervisor)
        agent_tools = [
            make_subagent_tool(
                name=name,
                description=agent.spec.description,
                agent=agent.compiled,
            )
            for name, agent in self._agents.items()
        ]

        # Build skill tools
        skill_tools = []
        for name, skill in self._skills.items():
            class _SkillInput(BaseModel):
                input: str

            st = StructuredTool.from_function(
                coroutine=lambda inp, s=skill: s(inp),
                name=name,
                description=skill.description,
                args_schema=_SkillInput,
            )
            skill_tools.append(st)

        # Build supervisor agent
        supervisor_spec = AgentSpec(
            name=f"{self.name}_supervisor",
            description=self.supervisor_prompt,
            system_prompt=self.supervisor_prompt,
            tools=[*agent_tools, *skill_tools],
            memory_backend=self.memory_backend,
            enable_guardrails=True,
            enable_tool_selector=True,
        )
        self._compiled = create_production_agent(supervisor_spec)
        return self

    async def ainvoke(
        self,
        user_message: str,
        *,
        thread_id: str,
        context: Any | None = None,
        user_id: str = "default",
    ) -> dict[str, Any]:
        if not self._compiled:
            self.build()
        return await self._compiled.ainvoke(
            user_message,
            thread_id=thread_id,
            context=context,
            user_id=user_id,
        )

    async def astream(
        self,
        user_message: str,
        *,
        thread_id: str,
        context: Any | None = None,
    ) -> AsyncIterator[Any]:
        if not self._compiled:
            self.build()
        async for chunk in self._compiled.astream(
            user_message, thread_id=thread_id, context=context
        ):
            yield chunk


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------


class LLMRouter:
    """
    Routes requests to agents or skills based on LLM classification.

    Usage::

        router = LLMRouter()
        router.add_route("research", research_agent, "For research and information gathering")
        router.add_route("code", code_agent, "For coding tasks")
        result = await router.route("How do I write a Python decorator?")
    """

    def __init__(self) -> None:
        self._routes: dict[str, tuple[Any, str]] = {}  # name → (handler, description)

    def add_route(self, name: str, handler: Any, description: str) -> LLMRouter:
        self._routes[name] = (handler, description)
        return self

    async def classify(self, user_input: str) -> str:
        """Return the name of the best matching route."""
        descriptions = {name: desc for name, (_, desc) in self._routes.items()}
        chain = build_router_chain(list(self._routes.keys()), descriptions=descriptions)
        result = await chain.ainvoke({"input": user_input, "history": []})
        return result.get("agent", next(iter(self._routes.keys())))

    async def route(
        self,
        user_input: str,
        *,
        thread_id: str,
        context: Any | None = None,
        user_id: str = "default",
    ) -> dict[str, Any]:
        """Classify and dispatch to the appropriate handler."""
        name = await self.classify(user_input)
        handler, _ = self._routes[name]
        logger.info("Router dispatching to: %s", name)

        if isinstance(handler, ProductionAgent):
            return await handler.ainvoke(
                user_input, thread_id=thread_id, context=context, user_id=user_id
            )
        if callable(handler):
            return await handler(user_input)
        raise TypeError(f"Unknown handler type: {type(handler)}")
