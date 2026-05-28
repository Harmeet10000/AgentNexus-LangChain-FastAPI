"""Configuration management for the Open Deep Search graph."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import Request
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings

if TYPE_CHECKING:
    from typing import Any



class Configuration(BaseModel):
    """Runtime configuration for Tavily-backed deep research.

    The graph intentionally does not own MCP configuration or provider-native
    web search. Those concerns live in ``app.shared.mcp`` and the shared
    Tavily service.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    max_structured_output_retries: int = Field(default=3, ge=1, le=10)
    allow_clarification: bool = True
    max_concurrent_research_units: int = Field(default=5, ge=1, le=20)
    max_researcher_iterations: int = Field(default=6, ge=1, le=10)
    max_react_tool_calls: int = Field(default=10, ge=1, le=30)

    summarization_model: str = Field(default_factory=lambda: get_settings().GEMINI_FLASH_MODEL)
    summarization_model_max_tokens: int = Field(default=8192, ge=1)
    max_content_length: int = Field(default=50_000, ge=1_000, le=200_000)

    research_model: str = Field(default_factory=lambda: get_settings().GEMINI_PRO_MODEL)
    research_model_max_tokens: int = Field(default=10_000, ge=1)

    compression_model: str = Field(default_factory=lambda: get_settings().GEMINI_PRO_MODEL)
    compression_model_max_tokens: int = Field(default=8192, ge=1)

    final_report_model: str = Field(default_factory=lambda: get_settings().GEMINI_PRO_MODEL)
    final_report_model_max_tokens: int = Field(default=10_000, ge=1)

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> Configuration:
        """Create a Configuration instance from RunnableConfig and environment."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})


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
            "tavily_http_client": getattr(request.app.state, "tavily_http_client", None),
            **(configurable or {}),
        },
        metadata=metadata or {},
        run_id=uuid4(),
        tags=["open_deep_search", *(tags or [])],
    )
