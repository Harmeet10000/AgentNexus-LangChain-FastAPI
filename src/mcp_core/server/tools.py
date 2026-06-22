from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from app.config import get_settings
from app.utils import NotFoundException, logger
from mcp_core.client.manager import get_mcp_client_manager
from mcp_core.common.metrics import observe_mcp_tool_invocation
from mcp_core.common.models import MCPToolCatalogEntry, MCPToolResponse
from mcp_core.server.factory import _server_name

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


_mcp_runtime: dict[str, FastAPI | None] = {"parent_app": None}


def bind_mcp_parent_app(app: FastAPI | None) -> None:
    _mcp_runtime["parent_app"] = app


def get_bound_mcp_parent_app() -> FastAPI | None:
    return _mcp_runtime["parent_app"]


def _truncate_payload(data: Any) -> Any:
    settings = get_settings()
    serialized = json.dumps(data, default=str)
    if len(serialized.encode("utf-8")) <= settings.MCP_MAX_RESULT_BYTES:
        return data

    truncated = serialized.encode("utf-8")[: settings.MCP_MAX_RESULT_BYTES].decode(
        "utf-8", errors="ignore"
    )
    return {
        "truncated": True,
        "preview": truncated,
        "max_bytes": settings.MCP_MAX_RESULT_BYTES,
    }


def _ok(data: Any, **metadata: Any) -> dict[str, Any]:
    response = MCPToolResponse(success=True, data=_truncate_payload(data), metadata=metadata)
    return response.model_dump(mode="json")


def _error(message: str, **metadata: Any) -> dict[str, Any]:
    response = MCPToolResponse(success=False, error=message, metadata=metadata)
    return response.model_dump(mode="json")


def _runtime_dependencies() -> dict[str, bool]:
    app = get_bound_mcp_parent_app()
    if app is None:
        return {
            "mounted": False,
            "httpx_client": False,
            "redis": False,
            "mongo_client": False,
            "db_engine": False,
            "neo4j_driver": False,
            "celery": False,
        }

    return {
        "mounted": True,
        "httpx_client": hasattr(app.state, "httpx_client"),
        "redis": hasattr(app.state, "redis"),
        "mongo_client": hasattr(app.state, "mongo_client"),
        "db_engine": hasattr(app.state, "db_engine"),
        "neo4j_driver": hasattr(app.state, "neo4j_driver"),
        "celery": getattr(app.state, "celery", None) is not None,
    }


def _tool_catalog() -> list[MCPToolCatalogEntry]:
    settings = get_settings()
    entries = [
        MCPToolCatalogEntry(
            id="tool:health_check",
            name="health_check",
            description="Return a lightweight MCP connectivity health payload.",
            read_only=True,
            tags=["health", "core"],
        ),
        MCPToolCatalogEntry(
            id="tool:readiness_check",
            name="readiness_check",
            description="Return dependency readiness for the mounted MCP runtime.",
            read_only=True,
            tags=["health", "readiness", "core"],
        ),
        MCPToolCatalogEntry(
            id="tool:get_server_metadata",
            name="get_server_metadata",
            description="Return MCP server metadata, transport configuration, and capability counts.",
            read_only=True,
            tags=["metadata", "core"],
        ),
        MCPToolCatalogEntry(
            id="tool:search",
            name="search",
            description="Search the curated MCP capability catalog and configured upstream servers.",
            read_only=True,
            tags=["catalog", "search"],
        ),
        MCPToolCatalogEntry(
            id="tool:fetch",
            name="fetch",
            description="Fetch a single capability or upstream-server record by identifier.",
            read_only=True,
            tags=["catalog", "fetch"],
        ),
        MCPToolCatalogEntry(
            id="tool:list_upstream_servers",
            name="list_upstream_servers",
            description="List approved upstream MCP servers and their health state.",
            read_only=True,
            tags=["upstream", "catalog"],
        ),
    ]

    return [
        entry
        for entry in entries
        if not settings.MCP_SERVER_ENABLED_TOOLS or entry.name in settings.MCP_SERVER_ENABLED_TOOLS
    ]


async def _timed_tool(
    tool_name: str,
    fn: Callable[[], Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    status = "success"
    try:
        result = fn()
        if hasattr(result, "__await__"):
            result = await result
    except NotFoundException as exc:
        status = "not_found"
        logger.bind(tool=tool_name, error=str(exc.detail)).warning("MCP tool failed")
        return _error(str(exc.detail))
    except Exception as exc:  # noqa: BLE001
        status = "error"
        logger.bind(tool=tool_name, error=str(exc)).exception("MCP tool failed")
        return _error("MCP tool execution failed", detail=str(exc))
    else:
        return result
    finally:
        observe_mcp_tool_invocation(
            tool_name=tool_name,
            status=status,
            duration_seconds=time.perf_counter() - start,
        )


def _register_tools(server: Any) -> None:
    settings = get_settings()
    catalog = {entry.name: entry for entry in _tool_catalog()}

    if "health_check" in catalog:

        @server.tool(annotations={"readOnlyHint": True})
        async def health_check() -> dict[str, Any]:
            async def _handler() -> dict[str, Any]:
                data = {
                    "status": "ok",
                    "service": settings.APP_NAME,
                    "version": settings.APP_VERSION,
                    "environment": settings.ENVIRONMENT,
                }
                return _ok(data)

            return await _timed_tool("health_check", _handler)

    if "readiness_check" in catalog:

        @server.tool(annotations={"readOnlyHint": True})
        async def readiness_check() -> dict[str, Any]:
            async def _handler() -> dict[str, Any]:
                dependencies = _runtime_dependencies()
                ready = dependencies["mounted"] and all(
                    dependencies[key]
                    for key in (
                        "httpx_client",
                        "redis",
                        "mongo_client",
                        "db_engine",
                        "neo4j_driver",
                    )
                )
                return _ok(
                    {
                        "status": "ready" if ready else "degraded",
                        "dependencies": dependencies,
                    }
                )

            return await _timed_tool("readiness_check", _handler)

    if "get_server_metadata" in catalog:

        @server.tool(annotations={"readOnlyHint": True})
        async def get_server_metadata() -> dict[str, Any]:
            async def _handler() -> dict[str, Any]:
                return _ok(
                    {
                        "name": _server_name(),
                        "app_name": settings.APP_NAME,
                        "version": settings.APP_VERSION,
                        "environment": settings.ENVIRONMENT,
                        "remote_http_enabled": settings.MCP_ENABLE_HTTP,
                        "local_stdio_enabled": settings.MCP_ENABLE_STDIO,
                        "http_path": settings.MCP_HTTP_PATH,
                        "http_transport": settings.MCP_HTTP_TRANSPORT,
                        "tool_count": len(catalog),
                        "tools": [entry.model_dump(mode="json") for entry in catalog.values()],
                    }
                )

            return await _timed_tool("get_server_metadata", _handler)

    if "search" in catalog:

        @server.tool(annotations={"readOnlyHint": True})
        async def search(
            query: str,
            limit: int = 5,
            offset: int = 0,
        ) -> dict[str, Any]:
            async def _handler() -> dict[str, Any]:
                query_lower = query.strip().lower()
                capability_matches = [
                    entry
                    for entry in catalog.values()
                    if query_lower in entry.name.lower() or query_lower in entry.description.lower()
                ]

                upstream_entries = await get_mcp_client_manager().discover_servers()
                upstream_matches = [
                    {
                        "id": f"upstream:{entry['name']}",
                        "name": entry["name"],
                        "description": entry.get("description", ""),
                        "kind": "upstream",
                    }
                    for entry in upstream_entries
                    if query_lower in entry["name"].lower()
                    or query_lower in entry.get("description", "").lower()
                ]

                results = [
                    {
                        "id": entry.id,
                        "name": entry.name,
                        "description": entry.description,
                        "kind": "tool",
                    }
                    for entry in capability_matches
                ] + upstream_matches

                safe_limit = max(1, min(limit, settings.MCP_MAX_PAGE_SIZE))
                safe_offset = max(0, offset)
                page = results[safe_offset : safe_offset + safe_limit]
                metadata = {
                    "limit": safe_limit,
                    "offset": safe_offset,
                    "total": len(results),
                    "has_more": safe_offset + safe_limit < len(results),
                }
                return _ok({"ids": [item["id"] for item in page], "results": page}, **metadata)

            return await _timed_tool("search", _handler)

    if "fetch" in catalog:

        @server.tool(annotations={"readOnlyHint": True})
        async def fetch(resource_id: str) -> dict[str, Any]:
            async def _handler() -> dict[str, Any]:
                if resource_id.startswith("tool:"):
                    tool_name = resource_id.split(":", 1)[1]
                    entry = catalog.get(tool_name)
                    if entry is None:
                        msg = "MCP tool"
                        raise NotFoundException(msg, tool_name)
                    return _ok(entry.model_dump(mode="json"))

                if resource_id.startswith("upstream:"):
                    server_name = resource_id.split(":", 1)[1]
                    return _ok(await get_mcp_client_manager().get_server_status(server_name))

                msg = "MCP catalog entry"
                raise NotFoundException(msg, resource_id)

            return await _timed_tool("fetch", _handler)

    if "list_upstream_servers" in catalog:

        @server.tool(annotations={"readOnlyHint": True})
        async def list_upstream_servers(limit: int = 10, offset: int = 0) -> dict[str, Any]:
            async def _handler() -> dict[str, Any]:
                servers = await get_mcp_client_manager().discover_servers()
                safe_limit = max(1, min(limit, settings.MCP_MAX_PAGE_SIZE))
                safe_offset = max(0, offset)
                page = servers[safe_offset : safe_offset + safe_limit]
                metadata = {
                    "limit": safe_limit,
                    "offset": safe_offset,
                    "total": len(servers),
                    "has_more": safe_offset + safe_limit < len(servers),
                }
                return _ok(page, **metadata)

            return await _timed_tool("list_upstream_servers", _handler)
