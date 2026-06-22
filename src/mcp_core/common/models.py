from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, Field, model_validator

from app.utils import ValidationException

if TYPE_CHECKING:
    from typing import Any


class MCPToolCatalogEntry(BaseModel):
    id: str
    name: str
    description: str
    read_only: bool = True
    tags: list[str] = Field(default_factory=list)


class MCPToolResponse(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MCPClientAuthMode(StrEnum):
    NONE = "none"
    BEARER = "bearer"


class MCPClientTransport(StrEnum):
    HTTP = "http"
    STDIO = "stdio"


class MCPClientServerConfig(BaseModel):
    name: str
    enabled: bool = False
    description: str = ""
    transport: MCPClientTransport = MCPClientTransport.HTTP
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    auth_mode: MCPClientAuthMode = MCPClientAuthMode.NONE
    bearer_token: str | None = None
    namespace: str | None = None
    enabled_tools: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    retry_attempts: int | None = None
    circuit_breaker_threshold: int | None = None
    circuit_breaker_cooldown_seconds: int | None = None

    @model_validator(mode="after")
    def validate_transport(self) -> MCPClientServerConfig:
        if self.transport == MCPClientTransport.HTTP and not self.url:
            msg = "HTTP transport requires 'url'"
            raise ValueError(msg)
        if self.transport == MCPClientTransport.STDIO and not self.command:
            msg = "STDIO transport requires 'command'"
            raise ValueError(msg)
        return self

    @property
    def namespace_prefix(self) -> str:
        return self.namespace or self.name

    @property
    def allowed_tools(self) -> set[str]:
        return set(self.enabled_tools)


class MCPClientCircuitState(BaseModel):
    failures: int = 0
    opened_until_epoch: float | None = None

    def is_open(self, now: float) -> bool:
        return self.opened_until_epoch is not None and self.opened_until_epoch > now


MCPHTTPTransport = Literal["http", "streamable-http", "sse"]
MCPTransport = Literal["stdio", "http", "streamable-http", "sse"]


def parse_mcp_http_transport(value: str) -> MCPHTTPTransport:
    allowed_values: tuple[MCPHTTPTransport, ...] = ("http", "streamable-http", "sse")
    if value not in allowed_values:
        msg = f"Unsupported MCP HTTP transport '{value}'. Expected one of: {', '.join(allowed_values)}"
        raise ValidationException(msg)
    return cast("MCPHTTPTransport", value)
