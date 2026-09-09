from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from app.utils import ValidationException

# The suite-wide conftest stubs MCP modules to keep unrelated tests isolated.
# Remove those stubs here because this test deliberately exercises the real
# configuration boundary.
for _module_name in tuple(sys.modules):
    if _module_name == "mcp_core" or _module_name.startswith("mcp_core."):
        del sys.modules[_module_name]

from mcp_core.client.settings import load_mcp_client_server_configs


def test_load_mcp_client_server_configs_returns_empty_for_blank_setting() -> None:
    with patch(
        "mcp_core.client.settings.get_settings",
        return_value=type("Settings", (), {"MCP_CLIENT_SERVER_CONFIGS": "  "})(),
    ):
        assert load_mcp_client_server_configs() == []


def test_load_mcp_client_server_configs_validates_json_array() -> None:
    raw = json.dumps(
        [
            {
                "name": "docs",
                "transport": "http",
                "url": "https://example.test/mcp",
            }
        ]
    )

    with patch(
        "mcp_core.client.settings.get_settings",
        return_value=type("Settings", (), {"MCP_CLIENT_SERVER_CONFIGS": raw})(),
    ):
        configs = load_mcp_client_server_configs()

    assert len(configs) == 1
    assert configs[0].name == "docs"
    assert configs[0].url == "https://example.test/mcp"


@pytest.mark.parametrize("raw", ["not json", json.dumps({"name": "docs"})])
def test_load_mcp_client_server_configs_rejects_invalid_json_shape(raw: str) -> None:
    with (
        patch(
            "mcp_core.client.settings.get_settings",
            return_value=type("Settings", (), {"MCP_CLIENT_SERVER_CONFIGS": raw})(),
        ),
        pytest.raises(
            ValidationException,
            match="MCP client server config validation failed",
        ),
    ):
        load_mcp_client_server_configs()


def test_load_mcp_client_server_configs_reports_model_errors() -> None:
    raw = json.dumps([{"name": "stdio-server", "transport": "stdio"}])

    with (
        patch(
            "mcp_core.client.settings.get_settings",
            return_value=type("Settings", (), {"MCP_CLIENT_SERVER_CONFIGS": raw})(),
        ),
        pytest.raises(ValidationException) as raised,
    ):
        load_mcp_client_server_configs()

    assert raised.value.data["errors"]
