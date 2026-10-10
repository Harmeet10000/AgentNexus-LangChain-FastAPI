"""LangChain tool for web crawling."""

import asyncio
from typing import Any, override

from langchain_core.tools import BaseTool


class CrawlUrlInput(BaseTool):
    """Input schema for crawl URL tool."""

    url: str
    extract_structured: bool = False
    schema_type: str | None = None
    custom_schema: dict[str, Any] | None = None
    summary: bool = False


class CrawlUrlTool(BaseTool):
    """Tool for crawling a URL and extracting content."""

    name: str = "crawl_url"
    description: str = """Crawl a specific URL and extract its content.
    Use this when you have a specific URL to fetch content from.
    Returns markdown content, optionally with structured data extraction or summary.
    """
    args_schema: type[CrawlUrlInput] = CrawlUrlInput

    @override
    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Synchronous fallback — langchain declares _run abstract; async-first tools bridge it."""

        return asyncio.run(self._arun(*args, **kwargs))

    @override
    async def _arun(
        self,
        url: str,
        extract_structured: bool = False,
        schema_type: str | None = None,
        custom_schema: dict[str, Any] | None = None,
        summary: bool = False,
    ) -> str:
        """Explain that immediate LLM crawling was intentionally removed."""
        _ = (url, extract_structured, schema_type, custom_schema, summary)
        return (
            "Immediate Crawl4AI tool execution is disabled. Use Tavily for bounded interactive "
            "web access, or submit a durable crawler job through the crawler API and retrieve "
            "its pages/chunks by crawl_id."
        )


def get_crawl_url_tool() -> CrawlUrlTool:
    return CrawlUrlTool()
