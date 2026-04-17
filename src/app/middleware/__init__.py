"""API middleware for error handling and request processing."""

from .global_exception_handler import global_exception_handler
from .server_middleware import (
    MetricsMiddleware,
    RequestStateLoggingMiddleware,
    build_fastapi_guard_config,
    get_metrics,
    initialize_fastapi_guard,
    metrics_registry,
    observe_mcp_client_call,
    observe_mcp_http_request,
    observe_mcp_tool_invocation,
    set_mcp_upstream_health,
)

__all__ = [
    "MetricsMiddleware",
    "RequestStateLoggingMiddleware",
    "build_fastapi_guard_config",
    "get_metrics",
    "global_exception_handler",
    "initialize_fastapi_guard",
    "metrics_registry",
    "observe_mcp_client_call",
    "observe_mcp_http_request",
    "observe_mcp_tool_invocation",
    "set_mcp_upstream_health",
]
