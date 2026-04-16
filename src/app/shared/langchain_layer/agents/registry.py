"""
Concrete agent definitions — wire everything together here.

This is where you define your actual agents using create_production_agent.
Each agent is a module-level singleton, lazy-initialised on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.tools import tool

# from app.shared.langgraph_layer.state import BaseContext, RichContext
from pydantic import BaseModel, Field

from ..prompts import SystemPromptParts
from .factory import AgentSpec, create_production_agent

# from agents.orchestration.supervisor import LLMRouter, MultiAgentSystem, Skill
from .tools.base import build_validation_error_handler
from .tools.shell import file_search, list_directory, read_file, shell_tool, write_file

if TYPE_CHECKING:
    from .factory import ProductionAgent

# ---------------------------------------------------------------------------
# Example: simple research tool
# ---------------------------------------------------------------------------


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query.")
    max_results: int = Field(5, description="Max results to return.")


@tool(
    args_schema=WebSearchInput,
    handle_tool_error=True,
    handle_validation_error=build_validation_error_handler(WebSearchInput),
)  # ty:ignore[no-matching-overload]
async def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web for current information. Returns a JSON list of results."""
    # Stub — replace with real search API (Serper, Tavily, etc.)
    return f'{{"query": "{query}", "results": ["Stub result 1", "Stub result 2"]}}'


# ---------------------------------------------------------------------------
# Context schemas
# ---------------------------------------------------------------------------


@dataclass
class CodeAgentContext(BaseContext):
    repo_path: str = "/tmp/repo"
    language: str = "python"
    allowed_extensions: list[str] = field(default_factory=lambda: [".py", ".js", ".ts"])


@dataclass
class ResearchAgentContext(BaseContext):
    depth: str = "standard"  # "quick" | "standard" | "deep"
    output_format: str = "markdown"


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------


class ResearchOutput(BaseModel):
    summary: str
    key_findings: list[str]
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


class CodeReviewOutput(BaseModel):
    issues: list[dict] = Field(default_factory=list)
    score: int = Field(..., ge=0, le=10)
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_research_agent() -> ProductionAgent:
    spec = AgentSpec(
        name="research_agent",
        description="Researches topics thoroughly using web search and synthesises findings.",
        model_name="gemini-2.0-flash",
        tools=[web_search_tool],
        system_prompt=SystemPromptParts(
            role=(
                "You are an expert research agent. Gather comprehensive, "
                "accurate information on any topic requested."
            ),
            capabilities=(
                "- Search the web for current information\n"
                "- Synthesise multiple sources into coherent summaries\n"
                "- Evaluate source credibility\n"
                "- Provide citations"
            ),
            output_format=(
                "Always structure your responses with:\n"
                "1. Executive Summary\n"
                "2. Key Findings\n"
                "3. Sources\n"
                "Confidence level: 0.0-1.0"
            ),
            constraints=(
                "- Do not fabricate sources or statistics.\n"
                "- Clearly distinguish between confirmed facts and inferences.\n"
                "- Always cite your sources."
            ),
        ),
        context_schema=ResearchAgentContext,
        response_format=ResearchOutput,
        enable_guardrails=True,
        enable_tool_selector=False,  # Only 1 tool — no selection needed
        max_tokens_before_summary=6000,
    )
    return create_production_agent(spec)


# ---------------------------------------------------------------------------
# Code agent
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_code_agent() -> ProductionAgent:
    spec = AgentSpec(
        name="code_agent",
        description="Writes, reviews, debugs, and explains code across multiple languages.",
        tools=[read_file, write_file, list_directory, file_search, shell_tool],
        system_prompt=SystemPromptParts(
            role=(
                "You are a senior software engineer and code assistant. "
                "You write clean, well-documented, production-grade code."
            ),
            capabilities=(
                "- Read and write files\n"
                "- Execute shell commands (when necessary)\n"
                "- Search through codebases\n"
                "- Debug and refactor code"
            ),
            output_format=(
                "Format code responses with:\n"
                "1. Brief explanation\n"
                "2. Code block(s) with language hints\n"
                "3. How to run/test\n"
                "4. Potential caveats"
            ),
            constraints=(
                "- Never execute commands that could cause data loss without confirmation.\n"
                "- Always validate file paths before writing.\n"
                "- Prefer reversible operations."
            ),
        ),
        context_schema=CodeAgentContext,
        response_format=None,  # Free-form text with code blocks
        enable_guardrails=True,
        enable_human_loop=True,
        human_loop_tools={"shell_tool": True, "write_file": True},
    )
    return create_production_agent(spec)


# ---------------------------------------------------------------------------
# General-purpose agent
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_general_agent() -> ProductionAgent:
    spec = AgentSpec(
        name="general_agent",
        description="A versatile assistant for general tasks and questions.",
        tools=[web_search_tool],
        context_schema=RichContext,
        enable_guardrails=True,
        max_tokens_before_summary=4000,
    )
    return create_production_agent(spec)


# ---------------------------------------------------------------------------
# Multi-agent system
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_multi_agent_system() -> MultiAgentSystem:
    system = MultiAgentSystem(name="production_mas")
    system.register_agent(get_research_agent())
    system.register_agent(get_code_agent())

    # Register a skill (cheap, no full agent)
    async def summarize_skill(text: str) -> str:
        from langchain_layer.chains import build_summarization_chain
        chain = build_summarization_chain(fast=True)
        return await chain.ainvoke({"input": text, "history": []})

    system.register_skill(Skill(
        name="summarize",
        description="Summarise any text quickly without full agent overhead.",
        fn=summarize_skill,
    ))

    system.build()
    return system


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    router = LLMRouter()
    router.add_route("research_agent", get_research_agent(), "Research topics, find information")
    router.add_route("code_agent", get_code_agent(), "Write, debug, or review code")
    router.add_route("general_agent", get_general_agent(), "General questions and tasks")
    return router
