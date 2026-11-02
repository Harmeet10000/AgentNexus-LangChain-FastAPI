"""LangChain tool for web search.

Egress-classified (P5): model-chosen text leaves the trust boundary, so the
tool requires the ``web.search`` grant like any other capability and refuses
without it. The tool resolves its server-owned runtime through the standard
``ToolRuntime`` injection seam (Decision 4 of ``tool-capability-catalog``):
LangChain injects it and hides it from the model-facing schema, so the model
chooses *what* to search but never *whether* it may.
"""

from __future__ import annotations

from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from pydantic import BaseModel, ConfigDict
from returns.result import Failure

from app.connections.tavily import get_tavily_client
from app.resilience.idempotency import ToolResult
from app.utils import ExternalServiceException

from .base import require_capability

#: Capability identifier enforced at execution (the egress gate, D4).
CAP_WEB_SEARCH = "web.search"


class WebSearchInput(BaseModel):
    """Input schema for web search tool.

    A plain ``BaseModel``, not a ``BaseTool``: this class is an argument
    *schema*, and inheriting the tool base made it a nameless, unusable tool
    in its own right while muddying the model-facing schema. The schema stays
    model-clean: the runtime context is injected, never an argument.

    Deliberately no ``extra="forbid"``: ToolNode injects the
    ``InjectedToolArg``-marked ``runtime`` kwarg into the call args before
    validating them against this schema, so forbidding extras rejects every
    ToolNode invocation (verified against langchain-core 1.7.0). The shell and
    filesystem schemas in ``shell.py`` rely on pydantic's default
    ``extra="ignore"`` for the same reason — they also take an injected
    ``runtime``. Extra model-supplied keys are ignored, never honored.
    """

    model_config = ConfigDict(extra="ignore")

    query: str
    max_results: int = 5


@tool(args_schema=WebSearchInput)
async def web_search(
    query: str,
    max_results: int = 5,
    runtime: Annotated[ToolRuntime | None, InjectedToolArg] = None,
) -> str | ToolResult:
    """
    Search the web for current information.

    Requires the ``web.search`` grant: without it the query is refused with a
    typed envelope and no text leaves the trust boundary.

    Args:
        query: Search query
        max_results: Maximum number of results (max 10)

    Returns:
        Formatted search results, or a typed refusal naming the capability.
    """
    try:
        require_capability(runtime, CAP_WEB_SEARCH)
    except PermissionError as exc:
        return ToolResult.fail(str(exc))

    max_results = min(max_results, 10)

    tavily = await get_tavily_client()

    result = await tavily.search(
        query=query,
        max_results=max_results,
        include_answer=True,
    )
    if isinstance(result, Failure):
        error = result.failure()
        raise ExternalServiceException(
            service="Tavily", detail=error.message, error_code=error.code.value
        )
    response = result.unwrap()

    parts = []

    if response.answer:
        parts.append(f"# Answer\n{response.answer}\n")

    parts.append("# Search Results\n")

    for i, result in enumerate(response.results, 1):
        parts.extend(
            [
                f"## {i}. {result.title}",
                f"**URL**: {result.url}",
                f"**Relevance**: {result.score:.2f}",
                f"\n{result.content}\n",
            ]
        )

    return "\n\n".join(parts)


def get_web_search_tool() -> BaseTool:
    """Get web search tool instance (the @tool-built ``web_search``)."""
    return web_search
