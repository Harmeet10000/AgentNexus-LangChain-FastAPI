from __future__ import annotations

import json
from pathlib import Path


def test_fastmcp_configuration_points_to_current_server_factory() -> None:
    config = json.loads(Path("fastmcp.json").read_text(encoding="utf-8"))
    source = config["source"]
    assert Path(source["path"]).is_file()
    assert source["entrypoint"] == "get_mcp_server"
