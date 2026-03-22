from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.config import get_settings
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
            raise ValueError("HTTP transport requires 'url'")
        if self.transport == MCPClientTransport.STDIO and not self.command:
            raise ValueError("STDIO transport requires 'command'")
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


def load_mcp_client_server_configs() -> list[MCPClientServerConfig]:
    raw = get_settings().MCP_CLIENT_SERVER_CONFIGS
    if not raw.strip():
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationException("MCP client server config JSON is invalid") from exc

    if not isinstance(payload, list):
        raise ValidationException("MCP client server config must be a JSON array")

    try:
        return [MCPClientServerConfig.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise ValidationException(
            "MCP client server config validation failed",
            data={"errors": exc.errors()},
        ) from exc


MCPHTTPTransport = Literal["http", "streamable-http", "sse"]


def parse_mcp_http_transport(value: str) -> MCPHTTPTransport:
    allowed_values: tuple[MCPHTTPTransport, ...] = ("http", "streamable-http", "sse")
    if value not in allowed_values:
        raise ValidationException(
            f"Unsupported MCP HTTP transport '{value}'. Expected one of: {', '.join(allowed_values)}"
        )
    return cast("MCPHTTPTransport", value)
