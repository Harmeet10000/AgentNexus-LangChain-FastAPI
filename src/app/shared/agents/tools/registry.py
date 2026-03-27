"""Tool registry for LangChain agents."""

from typing import Any

from .crawl import CrawlUrlTool, get_crawl_url_tool
from .web_search import WebSearchTool, get_web_search_tool


class ToolRegistry:
    """Registry for web search and crawl tools."""

    def __init__(self):
        self._tools: list[Any] = []

    def get_tools(self) -> list[Any]:
        """Get all registered tools."""
        if not self._tools:
            self._tools = [
                get_web_search_tool(),
                get_crawl_url_tool(),
            ]
        return self._tools

    def get_tool(self, name: str) -> Any:
        """Get a specific tool by name."""
        for tool in self.get_tools():
            if tool.name == name:
                return tool
        return None

    def get_search_tool(self) -> WebSearchTool:
        """Get the web search tool."""
        return get_web_search_tool()

    def get_crawl_tool(self) -> CrawlUrlTool:
        """Get the crawl URL tool."""
        return get_crawl_url_tool()


_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def get_all_tools() -> list[Any]:
    """Get all web search and crawl tools."""
    return get_tool_registry().get_tools()


def get_web_tools() -> list[Any]:
    """Get all web-related tools for agents."""
    return get_tool_registry().get_tools()
