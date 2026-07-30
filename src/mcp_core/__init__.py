__all__ = [
    "MCPClientManager",
    "MCPClientServerConfig",
    "MCPHTTPTransport",
    "MCPServerHandle",
    "MCPToolCatalogEntry",
    "MCPToolResponse",
    "bind_mcp_parent_app",
    "exchange_subject_token_for_mcp_token",
    "get_mcp_client_manager",
    "get_mcp_http_app",
    "get_mcp_server",
    "get_stored_mcp_tokens",
    "parse_mcp_http_transport",
    "run_mcp_server",
    "serve_mcp",
    "set_stored_mcp_tokens",
    "stop_mcp",
    "wrap_mcp_interaction_errors",
]


def __getattr__(name: str):
    if name in {
        "exchange_subject_token_for_mcp_token",
        "get_stored_mcp_tokens",
        "set_stored_mcp_tokens",
        "wrap_mcp_interaction_errors",
    }:
        from mcp_core.client.auth import (
            exchange_subject_token_for_mcp_token,
            get_stored_mcp_tokens,
            set_stored_mcp_tokens,
            wrap_mcp_interaction_errors,
        )

        return {
            "exchange_subject_token_for_mcp_token": exchange_subject_token_for_mcp_token,
            "get_stored_mcp_tokens": get_stored_mcp_tokens,
            "set_stored_mcp_tokens": set_stored_mcp_tokens,
            "wrap_mcp_interaction_errors": wrap_mcp_interaction_errors,
        }[name]

    if name in {"MCPClientManager", "get_mcp_client_manager"}:
        from mcp_core.client.manager import MCPClientManager, get_mcp_client_manager

        return MCPClientManager if name == "MCPClientManager" else get_mcp_client_manager

    if name in {
        "MCPClientServerConfig",
        "MCPHTTPTransport",
        "MCPToolCatalogEntry",
        "MCPToolResponse",
        "parse_mcp_http_transport",
    }:
        from mcp_core.common.models import (
            MCPClientServerConfig,
            MCPHTTPTransport,
            MCPToolCatalogEntry,
            MCPToolResponse,
            parse_mcp_http_transport,
        )

        return {
            "MCPClientServerConfig": MCPClientServerConfig,
            "MCPHTTPTransport": MCPHTTPTransport,
            "MCPToolCatalogEntry": MCPToolCatalogEntry,
            "MCPToolResponse": MCPToolResponse,
            "parse_mcp_http_transport": parse_mcp_http_transport,
        }[name]

    if name in {"MCPServerHandle", "serve_mcp", "stop_mcp"}:
        from mcp_core.lifespan_mcp import MCPServerHandle, serve_mcp, stop_mcp

        return {
            "MCPServerHandle": MCPServerHandle,
            "serve_mcp": serve_mcp,
            "stop_mcp": stop_mcp,
        }[name]

    if name in {"get_mcp_server"}:
        from mcp_core.server.factory import get_mcp_server

        return get_mcp_server

    if name in {"get_mcp_http_app", "run_mcp_server"}:
        from mcp_core.server.http import get_mcp_http_app, run_mcp_server

        return {
            "get_mcp_http_app": get_mcp_http_app,
            "run_mcp_server": run_mcp_server,
        }[name]

    if name in {"bind_mcp_parent_app"}:
        from mcp_core.server.tools import bind_mcp_parent_app

        return bind_mcp_parent_app

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
