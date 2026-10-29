"""MCP client manager dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.requests import HTTPConnection

if TYPE_CHECKING:
    from app.mcp import MCPClientManager


async def close_mcp_client_manager(manager: MCPClientManager | None) -> None:
    """Close all MCP upstream connections during lifespan shutdown."""
    if manager is not None:
        await manager.close()


def get_mcp_client_manager_dep(
    connection: HTTPConnection,
) -> MCPClientManager:
    """Dependency to inject MCP client manager from lifespan."""
    return connection.app.state.mcp_manager
