## Why

The MCP module at `src/app/shared/mcp/` is a flat monolithic directory mixing server-side (FastMCP registration, tool definitions, ASGI middleware) and client-side (upstream MCP connections, OAuth token exchange) concerns in the same namespace. It lives under `shared/` — a vestigial location from before the feature-driven structure was adopted. Meanwhile, FastMCP 3.x (already at `>=3.4.2` in dependencies) offers features the module does not use: composition (`mount()`/`create_proxy()`), resources (`@mcp.resource`), prompts (`@mcp.prompt`), Context DI, CodeMode for large tool catalogs, built-in pagination, and testing utilities. Splitting into `src/app/mcp/{server,client,common}/` and adopting these capabilities reduces coupling, enables domain-specific MCP tools, and closes the gap between what the library supports and what the module exposes.

## What Changes

### Restructure: `src/app/shared/mcp/` → `src/app/mcp/{server,client,common}/`

- **`src/app/mcp/server/`** — FastMCP server factory, tool registration, CLI entrypoint, ASGI middleware
- **`src/app/mcp/client/`** — Upstream MCP connection manager, OAuth token exchange/storage, LangGraph integration
- **`src/app/mcp/common/`** — Shared DTOs, types, settings
- Update all import sites to point to `app.mcp.*` instead of `app.shared.mcp.*`
- Remove `src/app/shared/mcp/` bundle

### Enhance: FastMCP 3.x capabilities

- **Server composition** — Use `mount()` and `create_proxy()` to attach upstream MCP servers as sub-servers instead of the custom `MCPClientManager` proxying layer
- **Resources** — Expose app config, feature status, and health data as `@mcp.resource` URIs
- **Prompts** — Define reusable prompt templates via `@mcp.prompt` for common agent interaction patterns
- **Context DI** — Use FastMCP's `Context` / `Depends` for dependency injection in tools (state, logging, progress)
- **CodeMode** — Enable for discovery-stage search so the catalog doesn't burn context on every LLM request
- **Telemetry** — Wire FastMCP's built-in telemetry hooks into the existing Prometheus registry
- **Testing** — Add `mcp/testing.py` with fixtures and helpers for FastMCP tool/resource testing
- **Lifespan** — Adopt FastMCP's composable lifespan hooks for server-level setup/teardown
- **Pagination** — Replace custom `_paginate()` with FastMCP's built-in `list_max_results` parameter

### Remove / Deprecate

- Remove custom `_paginate()`, `_truncate_payload()`, `_ok()`, `_error()` response helpers in favor of FastMCP-native patterns
- Remove `server.py` thin wrapper (fold into CLI entrypoint)
- **BREAKING**: `app.shared.mcp` imports removed — all callers must use `app.mcp.*`

## Capabilities

### New Capabilities
- `mcp-directory-restructure`: Split monolithic `src/app/shared/mcp/` into `src/app/mcp/{server,client,common}/` with clean separation of concerns
- `mcp-server-composition`: Leverage FastMCP `mount()` and `create_proxy()` for upstream server integration
- `mcp-server-resources`: Expose app data as MCP resources via `@mcp.resource`
- `mcp-server-prompts`: Expose reusable prompt templates via `@mcp.prompt`
- `mcp-server-codemode`: Enable CodeMode for context-efficient tool discovery
- `mcp-context-di`: Use FastMCP Context and Depends for tool dependency injection
- `mcp-telemetry`: Wire FastMCP telemetry hooks into existing Prometheus/OpenTelemetry stack
- `mcp-testing`: Testing infrastructure for FastMCP servers and tools
- `mcp-server-pagination`: Replace custom pagination with FastMCP built-in

### Modified Capabilities
- (none — no existing specs in `openspec/specs/` cover MCP)

## Impact

### Affected Code
- `src/app/shared/mcp/` — entirely replaced (7 files removed)
- `src/app/mcp/` — new module (estimated 12-15 files)
- `src/app/main.py` — update import path for `get_mcp_http_app`, `parse_mcp_http_transport`
- `src/app/connections/mcp.py` — update import path for `get_mcp_client_manager`
- `src/app/lifecycle/lifespan.py` — uncomment and update MCP client manager wiring
- `src/app/config/settings.py` — extract MCP settings into `src/app/mcp/common/settings.py`
- `tests/conftest.py` — update mocked paths from `app.shared.mcp` → `app.mcp`
- `src/app/examples/FastMCP-guide.md` — update all references

### Affected APIs
- No public HTTP API changes (same routes, same middleware protection)
- Internal API breaking change: `app.shared.mcp.*` → `app.mcp.*`

### Dependencies Added
- (none — `fastmcp>=3.4.2` already in `pyproject.toml`)

### Systems
- CI: all tests must pass after import migration
- CI: `uv run ruff check src/` and `uv run ty check src/` must pass
- CI: `openspec validate mcp-split-enhance` must pass
