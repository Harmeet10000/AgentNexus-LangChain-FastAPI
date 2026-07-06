from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from uvicorn.config import Config
from uvicorn.server import Server

from app.config import get_settings
from app.shared.otel import setup_otel
from mcp_core.server.http import get_mcp_http_app

if TYPE_CHECKING:
    from fastapi import FastAPI


@dataclass
class MCPServerHandle:
    server: Server
    task: asyncio.Task[None] = field(repr=False)


async def serve_mcp(parent_app: FastAPI) -> MCPServerHandle | None:
    settings = get_settings()
    if not settings.MCP_ENABLE_HTTP:
        return None

    if settings.OTEL_ENABLED:
        setup_otel(service_name="langchain-fastapi-mcp")

    mcp_app = get_mcp_http_app(parent_app=parent_app)
    config = Config(
        app=mcp_app,
        host=settings.MCP_HOST,
        port=settings.MCP_PORT,
        log_level=settings.MCP_LOG_LEVEL.lower(),
    )
    server = Server(config=config)
    task = asyncio.create_task(server.serve())
    return MCPServerHandle(server=server, task=task)


async def stop_mcp(handle: MCPServerHandle | None) -> None:
    if handle is None:
        return
    handle.server.should_exit = True
    handle.task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await handle.task
