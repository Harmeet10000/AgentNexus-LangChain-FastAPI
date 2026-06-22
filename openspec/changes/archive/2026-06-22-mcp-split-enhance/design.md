## Context

The MCP module was built before the project adopted feature-driven structure. It landed as `src/app/shared/mcp/` — a flat bucket with no internal separation between server, client, and common concerns. The current module has 7 files:

| File | Lines | Role |
|---|---|---|
| `registry.py` | 390 | Server factory, tool registration (6 tools), HTTP app builder, pagination/truncation helpers, standalone runner |
| `client.py` | 288 | Upstream MCP connection manager with circuit breaker, retries, semaphore |
| `security.py` | 202 | ASGI middleware: JWT auth, rate limiting, observability |
| `models.py` | 121 | DTOs for both server and client configs, circuit state, catalog entries |
| `best_practices.py` | 137 | OAuth token exchange, LangGraph token storage, error wrapping |
| `__init__.py` | 31 | Public API re-exports |
| `server.py` | 11 | Thin CLI entrypoint re-exporting from registry |

FastMCP 3.4.2+ (already installed) provides:
- **`mount()` / `create_proxy()`** — compose sub-servers, replacing custom proxying
- **`@mcp.resource`** — expose URI-addressable data (config, health, feature flags)
- **`@mcp.prompt`** — reusable prompt templates
- **`Context` / `Depends`** — dependency injection, state, logging, progress
- **CodeMode** — search-based tool discovery (avoids loading all tool schemas into context)
- **Built-in pagination** — `list_max_results` parameter on FastMCP constructor
- **Composable lifespans** — `Lifespan` class for server-level setup/teardown
- **Telemetry** — `FastMCP` telemetry hooks for metrics
- **Testing** — `FastMCP` client-with-server patterns for integration testing
- **Response transformation** — `ResponseLimitingMiddleware` (already used), transforms pipeline
- **Background tasks** — run async tasks in the MCP server context
- **Transport flexibility** — already using http/stdio/streamable-http/sse

## Goals / Non-Goals

**Goals:**
- Split monolithic module into `server/`, `client/`, `common/` sub-packages
- Adopt FastMCP `mount()`/`create_proxy()` for upstream server integration where it simplifies
- Add `@mcp.resource` and `@mcp.prompt` examples and patterns
- Enable CodeMode for context-efficient tool discovery
- Add FastMCP testing infrastructure
- Wire telemetry hooks
- Maintain backward compatibility at the HTTP API level (same routes, same auth)
- 100% coverage of existing functionality in new structure

**Non-Goals:**
- Convert the entire FastAPI REST API into MCP tools (intentionally curated)
- Remove the custom `MCPClientManager` entirely (keep for fine-grained allowlist/deny control that `create_proxy()` doesn't provide)
- Add write tools (deferred to a future change with stricter policy review)
- Expose all app state as resources (curated selection only)
- Replace the global exception handler or lifespan architecture outside MCP
- Implement OAuth provider flow for the MCP server (deferred)
- Remove `langchain-mcp-adapters` dependency (kept for agent integration)
- Support dynamic upstream registration by end users

## Decisions

### D1: Directory layout — `server/`, `client/`, `common/` under `src/app/mcp/`

**Decision:**
```
src/app/mcp/
├── __init__.py                # Re-exports public API
├── cli.py                     # Standalone CLI entrypoint (stdio, http modes)
├── server/
│   ├── __init__.py
│   ├── factory.py             # FastMCP server creation (was registry.py lines 343-390)
│   ├── tools.py               # Tool definitions (was registry.py lines 113-341)
│   ├── middleware.py          # ASGI middleware (was security.py)
│   └── http.py               # HTTP app builder, mounting (was registry.py lines 350-367)
├── client/
│   ├── __init__.py
│   ├── manager.py             # MCPClientManager (was client.py)
│   ├── auth.py                # Token exchange, storage, error wrapping (was best_practices.py)
│   └── settings.py            # MCPClientServerConfig loading (was models.py lines 86-108)
├── common/
│   ├── __init__.py
│   ├── models.py              # Shared DTOs (MCPToolCatalogEntry, MCPToolResponse, MCPClientCircuitState)
│   └── settings.py            # MCP-specific settings (extracted from config/settings.py)
└── testing.py                 # Test fixtures, helpers, and utilities
```

**Rationale:** Separates three independently deployable concerns. Server code can add tools without touching client code. Client code can add upstream providers without touching server middleware. Common models used by both avoid duplication.

**Alternatives considered:**
- *Keep flat structure under `app/shared/mcp/`*: already identified as problematic — rejected
- *Split into `app/server/mcp/` and `app/client/mcp/`*: too much directory nesting — rejected
- *Two separate top-level packages*: impractical because they share models — rejected

### D2: Server composition — use `create_proxy()` for trusted upstreams, keep `MCPClientManager` for controlled access

**Decision:** Two-tier upstream strategy:
- **Tier 1 (trusted):** Upstreams with no tool allowlist requirement mount via `create_proxy()`. These appear as sub-servers under the parent FastMCP. Their tools are auto-discovered.
- **Tier 2 (controlled):** Upstreams with per-tool allowlists continue through `MCPClientManager`. This gives explicit control over which tools are exposed.

**Rationale:** `create_proxy()` is simpler and gives native FastMCP tool discovery, but has no tool-level filtering. The custom manager provides allowlist enforcement that `create_proxy()` lacks. Keeping both gives the right tool for the right security posture.

**Alternatives considered:**
- *Replace `MCPClientManager` entirely with `create_proxy()` for all upstreams*: loses allowlist control — rejected
- *Build tool filtering on top of `create_proxy()`*: possible but adds custom complexity that duplicates the manager — rejected
- *Keep only `MCPClientManager`, ignore `create_proxy()`*: misses opportunity to simplify for trusted upstreams — rejected

### D3: Resources — add curated `@mcp.resource` URIs

**Decision:** Add resource endpoints for:
- `app://config` — current app settings (redacted secrets)
- `app://features` — feature flag status
- `app://health` — current health state
- `app://upstreams/{name}` — individual upstream MCP server status
- `mcp://catalog` — full tool/resource catalog

**Rationale:** Resources give MCP clients a standardized way to read data without calling tools. The `app://` scheme is FastMCP convention for application-level resources.

**Alternatives considered:**
- *Expose all DB data as resources*: massive security surface — rejected
- *Expose only health as a resource, keep everything else as tools*: reasonable, but config and feature flags are naturally read-only data, not operations — accepted compromise

### D4: Prompts — add 3-5 reusable prompt templates

**Decision:** Add prompts for:
- `explain-system` — explains the current system architecture and available MCP surface
- `diagnose-issue` — templates an incident diagnosis workflow
- `database-query` — safe SQL query pattern with guardrails
- `deploy-check` — pre-deployment checklist prompt
- `health-report` — summarizes system health from available tools/resources

**Rationale:** Prompts give LLM clients structured starting points for common tasks. They reduce hallucination by constraining the interaction pattern.

**Alternatives considered:**
- *Zero prompts (keep minimal surface)*: valid from YAGNI perspective, but prompts are zero-cost maintenance once defined — accepted as low-risk addition
- *Generate prompts from docstrings*: interesting but not worth the complexity — rejected

### D5: CodeMode — enable for all server modes

**Decision:** Enable CodeMode on the FastMCP server with a 3-stage discovery pipeline:
1. `search_tools()` — full-text search across tool names/descriptions
2. `get_tool_schema()` — returns JSON schema for a named tool
3. Code execution in sandboxed Python

**Rationale:** With the existing 6 tools and growing, CodeMode prevents LLM context bloat. The LLM only loads schemas for tools it actually plans to call. FastMCP 3.4+ has CodeMode built in — no custom implementation needed.

**Alternatives considered:**
- *Load all tools into every LLM request*: fine at 6 tools, won't scale to 20+ — adopted CodeMode proactively
- *Custom search endpoint*: redundant with CodeMode's built-in search — rejected

### D6: Pagination — replace custom `_paginate()` with FastMCP built-in

**Decision:** Use FastMCP's `list_max_results` constructor parameter instead of custom `_paginate()`. FastMCP 3.4+ handles pagination at the protocol level (adds `nextCursor` to list responses).

**Rationale:** Removing ~40 lines of custom code that the library already handles. FastMCP's built-in pagination is MCP-protocol-compliant; the custom version is not.

**Alternatives considered:**
- *Keep custom pagination and add FastMCP pagination on top*: duplication — rejected

### D7: Telemetry — wire FastMCP `on_tool_call` / `on_resource_read` hooks

**Decision:** Implement FastMCP's telemetry interface to increment existing Prometheus counters from `src/app/middleware/server_middleware.py`.

**Rationale:** Currently, metrics are incremented manually via `observe_mcp_tool_invocation()` inside `_timed_tool()`. FastMCP provides a standardized hook interface (`FastMCP.on_tool_call`, etc.) that fires for ALL tool calls — including those from mounted sub-servers or CodeMode.

**Alternatives considered:**
- *Keep manual instrumentation only*: misses sub-server and CodeMode calls — rejected
- *Add both*: double-counting risk — rejected in favor of hooks

### D8: Testing — dedicated `mcp/testing.py` with FastMCP test utilities

**Decision:** Add a `testing.py` module that provides:
- `MCPTestClient` — wraps `fastmcp.Client` to talk to a server in-process
- `pytest fixtures` — `mcp_server`, `mcp_client`, `mcp_tool_result`
- Helper to create a server with test-only tools for unit testing

**Rationale:** FastMCP supports in-process testing without starting a real server. Currently MCP has zero tests (mocked out in conftest). A dedicated testing module makes it easy for feature teams to add per-tool tests.

**Alternatives considered:**
- *Add tests inline in feature test files*: fine for feature-specific tests, but a shared `testing.py` avoids duplication — both approaches coexist

### D9: Settings — extract MCP settings into `src/app/mcp/common/settings.py`

**Decision:** Move all `MCP_*` settings from `src/app/config/settings.py` into a new `src/app/mcp/common/settings.py` as a Pydantic `BaseModel`. The main settings class composes it as a nested model.

**Rationale:** MCP has 25+ settings. Isolating them in the MCP package makes the module self-contained and easier to reason about. The main settings class references `mcp_settings: MCSettings` as a single field.

**Alternatives considered:**
- *Keep all settings in `config/settings.py`*: simpler for now but adds to that file's bloat (already 30+ fields) — rejected
- *Partial extraction (only client, not server)*: inconsistent — rejected

### D10: Import bridge — temporary compat shim for migration

**Decision:** Add `src/app/shared/mcp/__init__.py` compat shim that re-exports from `app.mcp.*` during migration, with a `DeprecationWarning`. Remove after one release cycle.

**Rationale:** Multiple files import from `app.shared.mcp`. A compat shim lets us land the restructure first, then migrate callers incrementally without a single atomic change.

**Alternatives considered:**
- *Single atomic migration of all callers*: riskier, harder to review — rejected
- *No shim, migrate everything at once*: possible but larger diff — accepted trade-off

## Risks / Trade-offs

- **[Compat shim debt]** The `app.shared.mcp` → `app.mcp` shim must be removed or it becomes permanent. **Mitigation:** Add a CI check (`# TODO: remove after mcp-split-enhance`) and a ticket to remove after next release.
- **[CodeMode sandbox security]** CodeMode executes LLM-written Python. **Mitigation:** FastMCP's sandbox runs in a restricted subprocess with no network/filesystem access by default. Keep this configuration unless a use case requires escalation.
- **[create_proxy() reliability]** Proxied upstreams are external dependencies. **Mitigation:** Keep the circuit breaker pattern for `MCPClientManager`-managed upstreams. For `create_proxy()` upstreams, rely on FastMCP's built-in error handling and timeouts.
- **[Test infrastructure newness]** `mcp/testing.py` introduces a new test pattern. **Mitigation:** Write one example test using it in the same PR, then document in the existing `src/app/examples/FastMCP-guide.md`.
- **[Settings extraction coupling]** Extracting MCP settings creates a dependency from `config/settings.py` → `mcp/common/settings.py`. **Mitigation:** The MCP settings model is standalone (no back-ref to `config/`), so the dependency direction is clean.
- **[Tool registration complexity]** Splitting tools from registry into `tools.py` may make it harder to see all tools at once. **Mitigation:** Keep `_register_tools()` as a single entry point that imports tool modules, so the factory still shows the full registration surface.
