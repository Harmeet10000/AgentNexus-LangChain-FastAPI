"""Domain-free agent tools: registry, web search, idempotency.

Domain tools live with their data owners (`features/documents/tools`,
`features/crawler/tools`) and are registered with role agents through
`app.retrieval.graphiti.registry` — never re-exported here.
"""

from app.resilience.idempotency import IdempotencyGuard, ToolResult

from .base import ToolRegistry
from .registry import (
    get_all_tools,
    get_tool_registry,
    get_web_tools,
    register_default_tools,
)
from .web_search import WebSearchTool, get_web_search_tool

__all__ = [
    "IdempotencyGuard",
    "ToolRegistry",
    "ToolResult",
    "WebSearchTool",
    "get_all_tools",
    "get_tool_registry",
    "get_web_search_tool",
    "get_web_tools",
    "register_default_tools",
]
