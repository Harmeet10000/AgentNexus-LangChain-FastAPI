"""Agent tools for web search, crawling and document processing."""



from .crawl import CrawlUrlTool, get_crawl_url_tool
from .get_obligation_chain import make_get_obligation_chain_tool
from .idempotency import IdempotencyGuard, ToolResult
from .query_knowledge_graph import make_query_knowledge_graph_tool
from .registry import (
    ToolRegistry,
    get_all_tools,
    get_tool_registry,
    get_web_tools,
)
from .retrieve_statute_section import make_retrieve_statute_section_tool
from .search_legal_precedents import make_search_legal_precedents_tool
from .web_search import WebSearchTool, get_web_search_tool

__all__ = [
    "CrawlUrlTool",
    "DocumentExtractionTools",
    "IdempotencyGuard",
    "ToolRegistry",
    "ToolResult",
    "WebSearchTool",
    "create_extraction_tools",
    "get_all_tools",
    "get_crawl_url_tool",
    "get_tool_registry",
    "get_web_search_tool",
    "get_web_tools",
    "make_get_obligation_chain_tool",
    "make_query_knowledge_graph_tool",
    "make_retrieve_statute_section_tool",
    "make_search_legal_precedents_tool",
]
