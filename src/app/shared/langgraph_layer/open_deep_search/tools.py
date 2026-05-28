"""LangChain tool wrapper for the deep research graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .graph import deep_researcher

if TYPE_CHECKING:
    from typing import Any

    import httpx
    from langchain_core.tools.base import BaseTool


class DeepResearchInput(BaseModel):
    """Input schema for the Agent Saul deep research tool."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Specific research question to investigate.")
    allow_clarification: bool = Field(
        default=False,
        description="Whether the research graph may stop and ask a clarifying question.",
    )
    max_concurrent_research_units: int = Field(default=3, ge=1, le=10)
    max_researcher_iterations: int = Field(default=4, ge=1, le=10)


class DeepResearchOutput(BaseModel):
    """Normalized output returned to an agent after deep research."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    final_report: str

    def to_agent_string(self) -> str:
        """Return a compact string for tool observations."""
        return self.final_report


def make_deep_research_tool(
    *,
    http_client: httpx.AsyncClient | None = None,
) -> BaseTool:
    """Build a ToolNode-compatible deep research tool."""

    async def run_deep_research(
        question: str,
        allow_clarification: bool = False,
        max_concurrent_research_units: int = 3,
        max_researcher_iterations: int = 4,
    ) -> str:
        result = await deep_researcher.ainvoke(
            cast("Any", {"messages": [HumanMessage(content=question)]}),
            config={
                "configurable": {
                    "allow_clarification": allow_clarification,
                    "max_concurrent_research_units": max_concurrent_research_units,
                    "max_researcher_iterations": max_researcher_iterations,
                    "httpx_client": http_client,
                    "tavily_http_client": http_client,
                }
            },
        )
        output = DeepResearchOutput(final_report=str(result.get("final_report", "")))
        return output.to_agent_string()

    return StructuredTool.from_function(
        coroutine=run_deep_research,
        name="deep_research",
        description=(
            "Run Tavily-backed deep web research and return a source-grounded report. "
            "Use for current external facts, market/legal background, and multi-source synthesis."
        ),
        args_schema=DeepResearchInput,
    )
