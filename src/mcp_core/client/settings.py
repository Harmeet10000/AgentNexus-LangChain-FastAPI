from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from app.config import get_settings
from app.utils import ValidationException
from mcp_core.common.models import MCPClientServerConfig

_MCP_CONFIG_ADAPTER = TypeAdapter(list[MCPClientServerConfig])


def load_mcp_client_server_configs() -> list[MCPClientServerConfig]:
    raw = get_settings().MCP_CLIENT_SERVER_CONFIGS
    if not raw.strip():
        return []

    try:
        return _MCP_CONFIG_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        msg = "MCP client server config validation failed"
        raise ValidationException(
            msg,
            data={"errors": exc.errors()},
        ) from exc
