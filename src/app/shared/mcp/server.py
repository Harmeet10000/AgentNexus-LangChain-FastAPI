from __future__ import annotations

from app.shared.mcp import registry

mcp = registry.get_mcp_server()
run_mcp_server = registry.run_mcp_server

__all__ = ["mcp", "run_mcp_server"]

if __name__ == "__main__":
    run_mcp_server()
