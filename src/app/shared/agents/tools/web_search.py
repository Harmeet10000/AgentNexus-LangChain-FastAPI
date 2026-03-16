"""LangChain tool for web search."""

from langchain_core.tools import BaseTool

from app.shared.services import get_tavily_client


class WebSearchInput(BaseTool):
    """Input schema for web search tool."""

    query: str
    max_results: int = 5


class WebSearchTool(BaseTool):
    """Tool for searching the web using Tavily."""

    name: str = "web_search"
    description: str = """Search the web for information.
    Use this when you need to find current information, facts, or answers.
    Returns relevant URLs with summaries and an AI-generated answer.
    """
    args_schema: type[WebSearchInput] = WebSearchInput

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
    ) -> str:
        """
        Run the web search tool.

        Args:
            query: Search query
            max_results: Maximum number of results (max 10)

        Returns:
            Formatted search results
        """
        max_results = min(max_results, 10)

        tavily = await get_tavily_client()

        response = await tavily.search(
            query=query,
            max_results=max_results,
            include_answer=True,
        )

        parts = []

        if response.answer:
            parts.append(f"# Answer\n{response.answer}\n")

        parts.append("# Search Results\n")

        for i, result in enumerate(response.results, 1):
            parts.append(f"## {i}. {result.title}")
            parts.append(f"**URL**: {result.url}")
            parts.append(f"**Relevance**: {result.score:.2f}")
            parts.append(f"\n{result.content}\n")

        return "\n\n".join(parts)

    def _run(self, *args, **kwargs) -> str:
        """Synchronous fallback."""
        import asyncio

        return asyncio.run(self._arun(*args, **kwargs))


def get_web_search_tool() -> WebSearchTool:
    """Get web search tool instance."""
    return WebSearchTool()
