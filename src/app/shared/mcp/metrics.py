from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from app.middleware.server_middleware import metrics_registry

mcp_http_requests_total = Counter(
    name="mcp_http_requests_total",
    documentation="Total inbound MCP HTTP requests",
    labelnames=["status_code", "path", "project"],
    registry=metrics_registry,
)

mcp_http_request_duration_seconds = Histogram(
    name="mcp_http_request_duration_seconds",
    documentation="Inbound MCP HTTP request duration in seconds",
    labelnames=["status_code", "path", "project"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=metrics_registry,
)

mcp_tool_calls_total = Counter(
    name="mcp_tool_calls_total",
    documentation="Total MCP tool calls handled by this server",
    labelnames=["tool", "status", "project"],
    registry=metrics_registry,
)

mcp_tool_call_duration_seconds = Histogram(
    name="mcp_tool_call_duration_seconds",
    documentation="MCP tool call duration in seconds",
    labelnames=["tool", "status", "project"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0),
    registry=metrics_registry,
)

mcp_client_calls_total = Counter(
    name="mcp_client_calls_total",
    documentation="Outbound MCP client tool calls",
    labelnames=["server", "tool", "status", "project"],
    registry=metrics_registry,
)

mcp_client_call_duration_seconds = Histogram(
    name="mcp_client_call_duration_seconds",
    documentation="Outbound MCP client tool call duration in seconds",
    labelnames=["server", "tool", "status", "project"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=metrics_registry,
)

mcp_upstream_server_health = Gauge(
    name="mcp_upstream_server_health",
    documentation="Health gauge for configured upstream MCP servers",
    labelnames=["server", "project"],
    registry=metrics_registry,
)

PROJECT = "langchain-fastapi"


def observe_mcp_http_request(path: str, status_code: int, duration_seconds: float) -> None:
    labels = {
        "status_code": str(status_code),
        "path": path,
        "project": PROJECT,
    }
    mcp_http_requests_total.labels(**labels).inc()
    mcp_http_request_duration_seconds.labels(**labels).observe(duration_seconds)


def observe_mcp_tool_invocation(tool_name: str, status: str, duration_seconds: float) -> None:
    labels = {"tool": tool_name, "status": status, "project": PROJECT}
    mcp_tool_calls_total.labels(**labels).inc()
    mcp_tool_call_duration_seconds.labels(**labels).observe(duration_seconds)


def observe_mcp_client_call(
    server_name: str,
    tool_name: str,
    status: str,
    duration_seconds: float,
) -> None:
    labels = {
        "server": server_name,
        "tool": tool_name,
        "status": status,
        "project": PROJECT,
    }
    mcp_client_calls_total.labels(**labels).inc()
    mcp_client_call_duration_seconds.labels(**labels).observe(duration_seconds)


def set_mcp_upstream_health(server_name: str, healthy: bool) -> None:
    mcp_upstream_server_health.labels(server=server_name, project=PROJECT).set(1 if healthy else 0)
