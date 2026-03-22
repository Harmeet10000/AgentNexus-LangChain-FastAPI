from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from functools import lru_cache
from typing import TYPE_CHECKING

from fastmcp import Client
from fastmcp.client.auth import BearerAuth

from app.config import get_settings
from app.shared.mcp.metrics import observe_mcp_client_call, set_mcp_upstream_health
from app.shared.mcp.models import (
    MCPClientCircuitState,
    load_mcp_client_server_configs,
)
from app.utils import ExternalServiceException, ServiceUnavailableException, logger

if TYPE_CHECKING:
    from typing import Any

    from app.shared.mcp.models import MCPClientServerConfig


class MCPClientManager:
    def __init__(self, server_configs: list[MCPClientServerConfig]) -> None:
        self._settings = get_settings()
        self._server_configs = {config.name: config for config in server_configs if config.enabled}
        self._clients: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {
            name: asyncio.Lock() for name in self._server_configs
        }
        self._circuits: dict[str, MCPClientCircuitState] = {
            name: MCPClientCircuitState() for name in self._server_configs
        }
        self._semaphore = asyncio.Semaphore(self._settings.MCP_CLIENT_MAX_CONCURRENCY)

    def configured_server_names(self) -> list[str]:
        return sorted(self._server_configs)

    def list_allowed_tools(self, server_name: str) -> list[str]:
        config = self._get_config(server_name)
        return sorted(config.allowed_tools)

    async def discover_servers(self) -> list[dict[str, Any]]:
        return [await self.get_server_status(name) for name in self.configured_server_names()]

    async def get_server_status(self, server_name: str) -> dict[str, Any]:
        config = self._get_config(server_name)
        circuit = self._circuits[server_name]
        now = time.time()
        state = "open" if circuit.is_open(now) else "closed"
        return {
            "name": config.name,
            "description": config.description,
            "transport": config.transport.value,
            "enabled": config.enabled,
            "namespace": config.namespace_prefix,
            "allowed_tools": sorted(config.allowed_tools),
            "circuit_state": state,
        }

    async def ping(self, server_name: str) -> bool:
        client = await self._connect(server_name)
        attempts = max(1, self._get_config(server_name).retry_attempts or self._settings.MCP_CLIENT_RETRY_ATTEMPTS)
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                result = await client.ping()
            except Exception as exc:
                last_error = exc
                continue
            else:
                set_mcp_upstream_health(server_name, True)
                self._record_success(server_name)
                return bool(result)

        self._record_failure(server_name, str(last_error) if last_error else "ping failed")
        raise ExternalServiceException(server_name, "Ping failed") from last_error

    async def list_remote_tools(self, server_name: str) -> list[dict[str, Any]]:
        client = await self._connect(server_name)
        allowed = self._get_config(server_name).allowed_tools
        tools = await client.list_tools()
        result = []
        for tool in tools:
            if allowed and tool.name not in allowed:
                continue
            result.append(
                {
                    "qualified_name": f"{self._get_config(server_name).namespace_prefix}_{tool.name}",
                    "server_name": server_name,
                    "tool_name": tool.name,
                    "description": getattr(tool, "description", ""),
                }
            )
        return result

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self._get_config(server_name)
        if config.allowed_tools and tool_name not in config.allowed_tools:
            raise ServiceUnavailableException(
                detail=f"Tool '{tool_name}' is not approved for upstream server '{server_name}'",
                error_code="MCP_UPSTREAM_TOOL_NOT_ALLOWED",
            )

        self._raise_if_circuit_open(server_name)
        client = await self._connect(server_name)
        start = time.perf_counter()
        status = "success"
        attempts = max(1, config.retry_attempts or self._settings.MCP_CLIENT_RETRY_ATTEMPTS)
        last_error: Exception | None = None

        async with self._semaphore:
            try:
                for _ in range(attempts):
                    try:
                        result = await client.call_tool(tool_name, arguments or {}, meta=meta)
                    except Exception as exc:
                        last_error = exc
                        continue
                    else:
                        self._record_success(server_name)
                        return self._normalize_tool_result(server_name, tool_name, result)
                status = "error"
                self._record_failure(server_name, str(last_error) if last_error else "tool call failed")
                raise ExternalServiceException(server_name, f"Tool call '{tool_name}' failed") from last_error
            finally:
                observe_mcp_client_call(
                    server_name=server_name,
                    tool_name=tool_name,
                    status=status,
                    duration_seconds=time.perf_counter() - start,
                )

    def get_tool_adapter(self, server_name: str, tool_name: str) -> dict[str, Any]:
        config = self._get_config(server_name)
        if config.allowed_tools and tool_name not in config.allowed_tools:
            raise ServiceUnavailableException(
                detail=f"Tool '{tool_name}' is not approved for upstream server '{server_name}'",
                error_code="MCP_UPSTREAM_TOOL_NOT_ALLOWED",
            )

        return {
            "qualified_name": f"{config.namespace_prefix}_{tool_name}",
            "server_name": server_name,
            "tool_name": tool_name,
            "callable": self.call_tool,
        }

    async def close(self) -> None:
        for name, client in list(self._clients.items()):
            with suppress(Exception):
                await client.__aexit__(None, None, None)
            set_mcp_upstream_health(name, False)
        self._clients.clear()

    def _get_config(self, server_name: str) -> MCPClientServerConfig:
        config = self._server_configs.get(server_name)
        if config is None:
            raise ServiceUnavailableException(
                detail=f"MCP upstream server '{server_name}' is not configured",
                error_code="MCP_UPSTREAM_NOT_CONFIGURED",
            )
        return config

    async def _connect(self, server_name: str) -> Any:
        if server_name in self._clients:
            return self._clients[server_name]

        lock = self._locks[server_name]
        async with lock:
            if server_name in self._clients:
                return self._clients[server_name]

            config = self._get_config(server_name)
            client = self._build_client(config)
            await client.__aenter__()
            self._clients[server_name] = client
            logger.bind(server=server_name).info("Connected MCP upstream client")
            return client

    def _build_client(self, config: MCPClientServerConfig) -> Any:
        auth: BearerAuth | None = None
        if config.auth_mode.value == "bearer":
            auth = BearerAuth(token=config.bearer_token or "")

        timeout = config.timeout_seconds or self._settings.MCP_CLIENT_DEFAULT_TIMEOUT_SECONDS

        if config.transport.value == "http":
            if config.url is None:
                raise ServiceUnavailableException(
                    detail=f"MCP upstream server '{config.name}' is missing a URL",
                    error_code="MCP_UPSTREAM_INVALID_CONFIG",
                )
            return Client(config.url, timeout=timeout, auth=auth)

        if config.command is None:
            raise ServiceUnavailableException(
                detail=f"MCP upstream server '{config.name}' is missing a command",
                error_code="MCP_UPSTREAM_INVALID_CONFIG",
            )
        source = {
            "command": config.command,
            "args": config.args,
            "env": config.env,
        }
        return Client(source, timeout=timeout, auth=auth)

    def _normalize_tool_result(self, server_name: str, tool_name: str, result: Any) -> dict[str, Any]:
        data = getattr(result, "data", None)
        content = getattr(result, "content", None)
        structured = getattr(result, "structured_content", None)
        payload = {
            "server_name": server_name,
            "tool_name": tool_name,
            "data": data,
            "content": content,
            "structured_content": structured,
            "raw_type": type(result).__name__,
        }
        serialized = str(payload)
        if len(serialized.encode("utf-8")) > self._settings.MCP_MAX_RESULT_BYTES:
            payload["content"] = None
            payload["structured_content"] = None
            payload["data"] = {
                "truncated": True,
                "raw_type": type(result).__name__,
                "max_bytes": self._settings.MCP_MAX_RESULT_BYTES,
            }
        return payload

    def _raise_if_circuit_open(self, server_name: str) -> None:
        circuit = self._circuits[server_name]
        if circuit.is_open(time.time()):
            raise ServiceUnavailableException(
                detail=f"Upstream MCP server '{server_name}' is temporarily unavailable",
                error_code="MCP_UPSTREAM_CIRCUIT_OPEN",
            )

    def _record_failure(self, server_name: str, error: str) -> None:
        config = self._get_config(server_name)
        circuit = self._circuits[server_name]
        circuit.failures += 1
        if circuit.failures >= (
            config.circuit_breaker_threshold
            or self._settings.MCP_CLIENT_CIRCUIT_BREAKER_THRESHOLD
        ):
            cooldown = (
                config.circuit_breaker_cooldown_seconds
                or self._settings.MCP_CLIENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS
            )
            circuit.opened_until_epoch = time.time() + cooldown
        set_mcp_upstream_health(server_name, False)
        logger.bind(server=server_name, error=error, failures=circuit.failures).warning(
            "MCP upstream failure recorded"
        )

    def _record_success(self, server_name: str) -> None:
        circuit = self._circuits[server_name]
        circuit.failures = 0
        circuit.opened_until_epoch = None
        set_mcp_upstream_health(server_name, True)


@lru_cache(maxsize=1)
def get_mcp_client_manager() -> MCPClientManager:
    if not get_settings().MCP_CLIENT_ENABLED:
        return MCPClientManager(server_configs=[])
    return MCPClientManager(server_configs=load_mcp_client_server_configs())
