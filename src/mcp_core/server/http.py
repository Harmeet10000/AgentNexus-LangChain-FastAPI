from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp.server.middleware.response_limiting import ResponseLimitingMiddleware

from app.config import get_settings

from .factory import get_mcp_server
from .middleware import build_mcp_http_middleware
from .tools import bind_mcp_parent_app

if TYPE_CHECKING:
    from typing import Any

    from fastapi import FastAPI

    from mcp_core.common.models import MCPHTTPTransport, MCPTransport


def get_mcp_http_app(
    *,
    parent_app: FastAPI | None = None,
    path: str = "/",
    transport: MCPHTTPTransport | None = None,
) -> Any:

    settings = get_settings()
    bind_mcp_parent_app(parent_app)
    middleware = [
        *build_mcp_http_middleware(parent_app=parent_app),
        ResponseLimitingMiddleware(max_size=settings.MCP_MAX_RESULT_BYTES),
    ]
    return get_mcp_server().http_app(
        path=path,
        transport=transport or settings.MCP_HTTP_TRANSPORT,
        middleware=middleware,
    )


def run_mcp_server(
    *,
    transport: MCPTransport | None = None,
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
) -> None:
    settings = get_settings()
    server = get_mcp_server()
    resolved_transport = transport or settings.MCP_RUN_TRANSPORT

    if resolved_transport == "stdio":
        server.run(transport="stdio", log_level=settings.MCP_LOG_LEVEL)
        return

    server.run(
        transport=resolved_transport,
        host=host or settings.MCP_HOST,
        port=port or settings.MCP_PORT,
        path=path or settings.MCP_HTTP_PATH,
        log_level=settings.MCP_LOG_LEVEL,
    )
