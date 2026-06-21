"""MCP client manager initialization and dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.requests import HTTPConnection

from app.shared.mcp import get_mcp_client_manager as _get_mcp_client_manager

if TYPE_CHECKING:
    from app.shared.mcp import MCPClientManager


def get_shared_mcp_client_manager() -> MCPClientManager:
    """Return a process-wide MCP client manager for non-request runtimes."""
    return _get_mcp_client_manager()


async def close_mcp_client_manager(manager: MCPClientManager | None) -> None:
    """Close all MCP upstream connections during lifespan shutdown."""
    if manager is not None:
        await manager.close()


def get_mcp_client_manager_dep(
    connection: HTTPConnection,
) -> MCPClientManager:
    """Dependency to inject MCP client manager from lifespan."""
    return connection.app.state.mcp_manager
