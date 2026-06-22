from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

_NOT_STARTED = "MCPTestClient not started. Use 'async with' context manager."


class MCPTestClient:
    """In-process MCP test client using FastMCP's client-with-server pattern."""

    def __init__(self, server: Any) -> None:
        self._server = server
        self._client: Any = None

    async def __aenter__(self) -> MCPTestClient:
        from fastmcp import Client  # noqa: PLC0415

        self._client = Client(self._server)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.__aexit__(*args)
            self._client = None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        if self._client is None:
            raise RuntimeError(_NOT_STARTED)
        return await self._client.call_tool(tool_name, arguments or {})

    async def read_resource(self, uri: str) -> Any:
        if self._client is None:
            raise RuntimeError(_NOT_STARTED)
        return await self._client.read_resource(uri)

    async def list_tools(self) -> list[Any]:
        if self._client is None:
            raise RuntimeError(_NOT_STARTED)
        return await self._client.list_tools()

    async def list_resources(self) -> list[Any]:
        if self._client is None:
            raise RuntimeError(_NOT_STARTED)
        return await self._client.list_resources()

    async def list_prompts(self) -> list[Any]:
        if self._client is None:
            raise RuntimeError(_NOT_STARTED)
        return await self._client.list_prompts()
