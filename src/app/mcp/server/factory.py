from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from app.config import get_settings

from .resources import register_resources
from .tools import _register_tools

if TYPE_CHECKING:
    from typing import Any


def _server_name() -> str:
    settings = get_settings()
    return settings.MCP_SERVER_NAME or f"{settings.APP_NAME} MCP"


def _instructions() -> str:
    settings = get_settings()
    return (
        f"{settings.APP_NAME} curated MCP server. "
        "Use exposed tools only. Prefer read-only inspection tools before expensive operations."
    )


@cache
def get_mcp_server() -> Any:
    settings = get_settings()
    server = FastMCP(
        name=_server_name(),
        instructions=_instructions(),
        list_page_size=settings.MCP_MAX_PAGE_SIZE,
    )
    _register_tools(server)
    register_resources(server)
    from .prompts import register_prompts

    register_prompts(server)
    return server
