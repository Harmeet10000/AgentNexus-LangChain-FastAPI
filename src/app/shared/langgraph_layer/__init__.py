"""Open Deep Search package exports."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import Request
from langchain_core.runnables import RunnableConfig

from .open_deep_search import Configuration, MCPConfig, SearchAPI, deep_researcher

if TYPE_CHECKING:
    from typing import Any


def build_open_deep_search_config(
    request: Request,
    *,
    thread_id: str,
    configurable: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> RunnableConfig:
    """Build graph config with the lifespan-owned HTTPX client attached."""
    return RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "httpx_client": request.app.state.httpx_client,
            **(configurable or {}),
        },
        metadata=metadata or {},
        run_id=uuid4(),
        tags=["open_deep_search", *(tags or [])],
    )


__all__ = [
    "Configuration",
    "MCPConfig",
    "SearchAPI",
    "build_open_deep_search_config",
    "deep_researcher",
]
