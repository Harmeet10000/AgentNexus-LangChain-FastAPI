from __future__ import annotations

import json
from typing import TYPE_CHECKING

from app.config import get_settings
from mcp_core.client.manager import get_mcp_client_manager
from mcp_core.server.tools import _tool_catalog, get_bound_mcp_parent_app

if TYPE_CHECKING:
    from typing import Any


MCP_MAX_RESULT_BYTES = 524288


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return "***"
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _truncate(data: Any) -> Any:
    serialized = json.dumps(data, default=str)
    if len(serialized.encode("utf-8")) <= MCP_MAX_RESULT_BYTES:
        return data
    return {"truncated": True, "max_bytes": MCP_MAX_RESULT_BYTES}


async def _get_config() -> dict[str, Any]:
    settings = get_settings()
    raw = settings.model_dump(mode="json")
    sensitive_keys = {
        "JWT_SECRET_KEY",
        "NEO4J_PASSWORD",
        "GEMINI_API_KEY",
        "RESEND_API_KEY",
        "OAUTH_STATE_SECRET",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "TAVILY_API_KEY",
        "PINECONE_API_KEY",
        "LANGEXTRACT_API_KEY",
        "LANGSMITH_API_KEY",
        "REDIS_PASSWORD",
        "RABBITMQ_DEFAULT_PASS",
        "GOOGLE_CLIENT_SECRET",
    }
    for key in sensitive_keys:
        if key in raw:
            raw[key] = "***"
    return _truncate(raw)


async def _get_features() -> dict[str, Any]:
    settings = get_settings()
    features = {
        "mcp_enabled": settings.MCP_ENABLE_HTTP,
        "mcp_stdio": settings.MCP_ENABLE_STDIO,
        "mcp_auth_required": settings.MCP_REQUIRE_AUTH,
        "rate_limiting": settings.RATE_LIMIT_ENABLED,
        "crawl4ai": bool(settings.CRAWL4AI_HEADLESS),
    }
    return _truncate(features)


async def _get_health() -> dict[str, Any]:
    app = get_bound_mcp_parent_app()
    if app is None:
        return {"status": "unknown", "dependencies": {}}

    deps = {
        "mounted": True,
        "httpx_client": hasattr(app.state, "httpx_client"),
        "redis": hasattr(app.state, "redis"),
        "mongo_client": hasattr(app.state, "mongo_client"),
        "db_engine": hasattr(app.state, "db_engine"),
        "neo4j_driver": hasattr(app.state, "neo4j_driver"),
        "celery": getattr(app.state, "celery", None) is not None,
    }
    ready = all(deps[key] for key in ("httpx_client", "redis", "mongo_client", "db_engine"))
    return _truncate({"status": "ready" if ready else "degraded", "dependencies": deps})


async def _get_upstream_status(server_name: str) -> dict[str, Any]:
    return await get_mcp_client_manager().get_server_status(server_name)


async def _get_catalog() -> dict[str, Any]:
    tools = [entry.model_dump(mode="json") for entry in _tool_catalog()]
    return _truncate({"tools": tools})


def register_resources(server: Any) -> None:
    @server.resource("app://config")
    async def config_resource() -> dict[str, Any]:
        return await _get_config()

    @server.resource("app://features")
    async def features_resource() -> dict[str, Any]:
        return await _get_features()

    @server.resource("app://health")
    async def health_resource() -> dict[str, Any]:
        return await _get_health()

    @server.resource("app://upstreams/{server_name}")
    async def upstream_resource(server_name: str) -> dict[str, Any]:
        return await _get_upstream_status(server_name)

    @server.resource("mcp://catalog")
    async def catalog_resource() -> dict[str, Any]:
        return await _get_catalog()
