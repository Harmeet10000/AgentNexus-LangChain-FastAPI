## 0. Scaffold & Dependency

- [x] 0.1 Create `src/mcp_core/` directory tree (`server/`, `client/`, `common/`)
- [x] 0.2 Create `__init__.py` with public API re-exports
- [x] 0.3 Create compat shim at `src/app/shared/mcp/__init__.py` with deprecation warning (re-exports from `mcp_core`)
- [x] 0.4 Review `fastmcp>=3.4.2` capabilities — confirmed `3.4.2`

## 1. Directory Restructure — Server Module

- [x] 1.1 Create `src/mcp_core/server/__init__.py`
- [x] 1.2 Create `src/mcp_core/server/factory.py` — `get_mcp_server()`, `_server_name()`, `_instructions()` from `registry.py`
- [x] 1.3 Create `src/mcp_core/server/tools.py` — `_tool_catalog()`, `_register_tools()`, all 6 tool definitions from `registry.py`
- [x] 1.4 Create `src/mcp_core/server/middleware.py` — `MCPAuthMiddleware`, `MCPRateLimitMiddleware`, `MCPObservabilityMiddleware`, `build_mcp_http_middleware()` from `security.py`
- [x] 1.5 Create `src/mcp_core/server/http.py` — `get_mcp_http_app()`, `run_mcp_server()` from `registry.py`
- [x] 1.6 Create `src/mcp_core/cli.py` — standalone entrypoint (was `server.py`)
- [x] 1.7 Move `ResponseLimitingMiddleware` import, `_truncate_payload()`, `_ok()`, `_error()` helpers into server utilities

## 2. Directory Restructure — Client Module

- [x] 2.1 Create `src/mcp_core/client/__init__.py`
- [x] 2.2 Create `src/mcp_core/client/manager.py` — `MCPClientManager` class from `client.py`
- [x] 2.3 Create `src/mcp_core/client/auth.py` — `exchange_subject_token_for_mcp_token()`, `get_stored_mcp_tokens()`, `set_stored_mcp_tokens()`, `wrap_mcp_interaction_errors()` from `best_practices.py`
- [x] 2.4 Create `src/mcp_core/client/settings.py` — `load_mcp_client_server_configs()` from `models.py`

## 3. Directory Restructure — Common Module

- [x] 3.1 Create `src/mcp_core/common/__init__.py`
- [x] 3.2 Create `src/mcp_core/common/models.py` — `MCPToolCatalogEntry`, `MCPToolResponse`, `MCPClientCircuitState`, `MCPClientAuthMode`, `MCPClientTransport`, `MCPClientServerConfig`, `MCPHTTPTransport`, `MCPTransport`, `parse_mcp_http_transport()` from `models.py`
- [ ] 3.3 Create `src/mcp_core/common/settings.py` — extract all `MCP_*` settings into `MCPSettings` Pydantic model
- [ ] 3.4 Integrate `MCPSettings` into `src/app/config/settings.py` as `mcp_settings: MCPSettings`
- [ ] 3.5 Update all `get_settings().MCP_*` references across MCP module to use nested access

## 4. Import Path Migration

- [x] 4.1 Update `src/app/main.py` — import from `mcp_core` instead of `app.shared.mcp`
- [x] 4.2 Update `src/app/connections/mcp.py` — import from `mcp_core` instead of `app.shared.mcp`
- [x] 4.3 Update `src/app/lifecycle/lifespan.py` — uncomment MCP client manager wiring, update import
- [x] 4.4 Update `tests/conftest.py` — mock `mcp_core` and `mcp_core.server.middleware` instead of `app.shared.mcp.*`
- [x] 4.5 Create compat shim at `src/app/shared/mcp/__init__.py` re-exporting all symbols from `mcp_core` with `DeprecationWarning`
- [ ] 4.6 Update `src/app/examples/FastMCP-guide.md` — all references to new paths
- [x] 4.7 Remove `src/app/shared/mcp/` old files (done in section 5)
- [x] 4.8 Run `uv run ruff check src/mcp_core/` — 0 errors
- [x] 4.9 Run `uv run ty check src/mcp_core/` — 0 errors

## 5. MCP Middleware & Observability — Migrate to mcp_core

- [x] 5.1 Audit old `src/app/shared/mcp/` for gaps vs `mcp_core/` — found: missing `MCPTransport` type, missing module docstring in `client/auth.py`, dead guard in `tools.py:get_server_metadata`
- [x] 5.2 Fix gaps: restore module docstring in `client/auth.py`
- [x] 5.3 Fix gaps: add `MCPTransport` type alias to `common/models.py`
- [x] 5.4 Fix gaps: use `MCPTransport` and `MCPHTTPTransport` in `server/http.py` instead of `str`
- [x] 5.5 Fix gaps: remove dead guard `_server_name() if ... else ""` in `tools.py:get_server_metadata`
- [x] 5.6 Create `mcp_core/common/metrics.py` — move 4 observer functions + 7 Prometheus metric definitions from `app/middleware/server_middleware.py`
- [x] 5.7 Update `mcp_core/server/tools.py` — import `observe_mcp_tool_invocation` from `mcp_core.common.metrics`
- [x] 5.8 Update `mcp_core/server/middleware.py` — import `observe_mcp_http_request` from `mcp_core.common.metrics`
- [x] 5.9 Update `mcp_core/client/manager.py` — import `observe_mcp_client_call`, `set_mcp_upstream_health` from `mcp_core.common.metrics`
- [x] 5.10 Remove MCP metric definitions and observer functions from `app/middleware/server_middleware.py`
- [x] 5.11 Remove MCP observer re-exports from `app/middleware/__init__.py`
- [x] 5.12 Delete old `src/app/shared/mcp/` directory (all gaps fixed, compat shim removed)
- [x] 5.13 Run `uv run ruff check src/mcp_core/` — 0 errors
- [x] 5.14 Run `uv run ty check src/mcp_core/` — 0 errors

## 6. Server Composition — create_proxy Integration

- [ ] 6.1 Define upstream proxy configuration in `MCPClientServerConfig` — add `proxy_enabled: bool = False` field
- [ ] 6.2 Implement proxy mounting in `factory.py`: iterate configs, call `create_proxy()` for proxy-enabled upstreams, mount via `mcp.mount()`
- [ ] 6.3 Update `list_upstream_servers` tool to report `connection_type` per upstream
- [ ] 6.4 Add timeout configuration for proxy-mounted upstreams using FastMCP client timeout
- [ ] 6.5 Write unit test: proxy and manager upstreams coexist
- [ ] 6.6 Run `uv run ruff check src/mcp_core/` and `uv run ty check src/mcp_core/`

## 7. Resources — @mcp.resource Definitions

- [x] 7.1 Create `src/mcp_core/server/resources.py` with resource definitions
- [x] 7.2 Add `app://config` resource — app settings with secret redaction
- [x] 7.3 Add `app://features` resource — feature flag status
- [x] 7.4 Add `app://health` resource — dependency health state
- [x] 7.5 Add `app://upstreams/{server_name}` templated resource — per-upstream status
- [x] 7.6 Add `mcp://catalog` resource — full tool/resource/prompt catalog
- [x] 7.7 Register resources in `factory.py` (via `register_resources()` call)
- [ ] 7.8 Write test: each resource returns expected shape
- [x] 7.9 Run `uv run ruff check src/mcp_core/` — 0 errors
- [x] 7.10 Run `uv run ty check src/mcp_core/` — 0 errors

## 8. Prompts — @mcp.prompt Definitions

- [x] 8.1 Create `src/mcp_core/server/prompts.py` with prompt definitions
- [x] 8.2 Add `explain_system` prompt
- [x] 8.3 Add `diagnose_issue` prompt
- [x] 8.4 Add `database_query` prompt
- [x] 8.5 Add `deploy_check` prompt
- [x] 8.6 Add `health_report` prompt
- [x] 8.7 Register prompts in `factory.py`
- [ ] 8.8 Write test: each prompt returns expected template
- [x] 8.9 Run `uv run ruff check src/mcp_core/` — 0 errors
- [x] 8.10 Run `uv run ty check src/mcp_core/` — 0 errors

## 9. CodeMode

- [x] 9.1 Enable CodeMode via `list_max_results` in FastMCP constructor
- [x] 9.2 Configure with `list_max_results=settings.MCP_MAX_PAGE_SIZE`
- [ ] 9.3 Verify `search_tools` and `get_tool_schema` appear in tool list
- [ ] 9.4 Verify curated tools coexist with CodeMode tools
- [ ] 9.5 Write test: CodeMode discovery tools work
- [x] 9.6 Run `uv run ruff check src/mcp_core/` — 0 errors

## 10. Context Dependency Injection

- [ ] 10.1 Update tool signatures to use `ctx: Context` parameter
- [ ] 10.2 Replace `logger.bind().info()` with `ctx.info()` inside tool handlers
- [ ] 10.3 Add `ctx.report_progress()` calls in paginated/batch operations
- [ ] 10.4 Define `Depends`-based dependency functions in `src/mcp_core/server/deps.py`
- [ ] 10.5 Wire settings dependency via `Depends(get_settings)`
- [ ] 10.6 Remove module-level logger calls from tool handlers (keep logger for non-request contexts like middleware)
- [ ] 10.7 Write test: Context state persists across tool calls in same session
- [ ] 10.8 Run `uv run ruff check src/mcp_core/` and `uv run ty check src/mcp_core/`

## 11. Telemetry Hooks

- [ ] 11.1 Define telemetry hook class or callbacks in `src/mcp_core/server/telemetry.py`
- [ ] 11.2 Wire `on_tool_call` hook to increment `mcp_tool_calls_total` and `mcp_tool_call_duration_seconds`
- [ ] 11.3 Wire `on_resource_read` hook to appropriate metrics
- [ ] 11.4 Wire `on_prompt_executed` hook to appropriate metrics
- [ ] 11.5 Register telemetry hooks on FastMCP server instance in `factory.py`
- [ ] 11.6 Remove `observe_mcp_tool_invocation()` calls from `_timed_tool()`
- [ ] 11.7 Verify HTTP-level metrics (`mcp_http_requests_total`, `mcp_http_request_duration_seconds`) still work via middleware
- [ ] 11.8 Run `uv run ruff check src/mcp_core/` and `uv run ty check src/mcp_core/`

## 12. Pagination

- [x] 12.1 Add `list_max_results=get_settings().MCP_MAX_PAGE_SIZE` to `FastMCP()` constructor in `factory.py`
- [x] 12.2 Remove `_paginate()` function and `_catalog_by_name()` internal helpers
- [x] 12.3 Update `search` tool — inline pagination, removed `_paginate()` call
- [x] 12.4 Update `list_upstream_servers` tool — inline pagination, removed `_paginate()` call
- [x] 12.5 Keep `limit`/`offset` parameters on tool signatures for backward compat
- [ ] 12.6 Write test: paginated tool responses include `nextCursor` (or equivalent pagination metadata)
- [x] 12.7 Run `uv run ruff check src/mcp_core/` — 0 errors

## 13. Testing Infrastructure

- [x] 13.1 Create `src/mcp_core/testing.py` with `MCPTestClient` class
- [x] 13.2 Implement `MCPTestClient.__init__()` — accept FastMCP server, create in-process Client
- [x] 13.3 Implement async context manager protocol for `MCPTestClient`
- [ ] 13.4 Add `pytest` fixtures in `tests/conftest.py` or `tests/fixtures/mcp.py`
- [ ] 13.5 Add test tool helper function
- [ ] 13.6 Write example test: `test_mcp_health_check` using `MCPTestClient`
- [ ] 13.7 Write example test: `test_mcp_resource_config` using `MCPTestClient`
- [ ] 13.8 Write example test: `test_mcp_composition_proxy_and_manager` using `MCPTestClient`
- [ ] 13.9 Verify all MCP tests pass with `uv run pytest tests/ -v -k mcp`
- [ ] 13.10 Verify existing tests still pass with mock update

## 14. Cleanup & Final Verification

- [x] 14.1 Old `src/app/shared/mcp/` deleted (compat shim removed)
- [ ] 14.2 Run full test suite: `uv run pytest tests/ -v`
- [ ] 14.3 Run `uv run ruff check src/` — zero new issues
- [ ] 14.4 Run `uv run ty check src/` — zero new issues
- [x] 14.5 Run `openspec validate mcp-split-enhance`
- [ ] 14.6 Update `src/app/examples/FastMCP-guide.md` final pass
- [ ] 14.7 Archive change: `openspec archive mcp-split-enhance`
