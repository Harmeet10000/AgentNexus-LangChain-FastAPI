from __future__ import annotations

import json

from pydantic import ValidationError

from app.config import get_settings
from app.utils import ValidationException
from mcp_core.common.models import MCPClientServerConfig


def load_mcp_client_server_configs() -> list[MCPClientServerConfig]:
    raw = get_settings().MCP_CLIENT_SERVER_CONFIGS
    if not raw.strip():
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = "MCP client server config JSON is invalid"
        raise ValidationException(msg) from exc

    if not isinstance(payload, list):
        msg = "MCP client server config must be a JSON array"
        raise ValidationException(msg)

    try:
        return [MCPClientServerConfig.model_validate(item) for item in payload]
    except ValidationError as exc:
        msg = "MCP client server config validation failed"
        raise ValidationException(
            msg,
            data={"errors": exc.errors()},
        ) from exc
