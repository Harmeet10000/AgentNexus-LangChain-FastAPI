# FastMCP Guide

This project includes a production-oriented FastMCP integration under `src/app/shared/mcp`.

The implementation is designed around two deployment modes from the same codebase:
- local `stdio` for developer-tool style clients
- remote HTTP mounted into the main FastAPI application at `MCP_HTTP_PATH`

It also includes an internal MCP client manager for approved third-party MCP servers.

## What Is Implemented

### 1. MCP server foundation

The MCP server lives in:
- `src/app/shared/mcp/registry.py`
- `src/app/shared/mcp/server.py`

What it provides:
- `get_mcp_server()` to build a cached FastMCP server
- `get_mcp_http_app()` to build a mounted ASGI app
- `run_mcp_server()` to run the server in `stdio` or HTTP mode
- `server.py` as the CLI/server entrypoint for FastMCP tooling

The server is intentionally curated. It does not mirror the whole FastAPI API surface.

Current exposed MCP tools:
- `health_check`
- `readiness_check`
- `get_server_metadata`
- `search`
- `fetch`
- `list_upstream_servers`

These tools are read-only and meant to be safe defaults for production bootstrapping.

### 2. Remote HTTP integration

The MCP server is mounted from `src/app/main.py`.

Important behavior:
- MCP is mounted only when `MCP_ENABLE_HTTP=true`
- the mounted app uses `get_mcp_http_app(parent_app=app, path="/", ...)`
- FastMCP lifespan is combined with the existing FastAPI lifespan using `combine_lifespans(...)`
- the mounted endpoint is available at `settings.MCP_HTTP_PATH`

This matters because FastMCP’s HTTP app has its own session/runtime initialization and cannot safely replace the app lifespan.

### 3. MCP security and protection

The HTTP MCP middleware lives in `src/app/shared/mcp/security.py`.

Current protections:
- JWT bearer authentication for remote HTTP requests
- Redis-backed rate limiting per subject and path
- MCP request logging with correlation IDs
- MCP HTTP Prometheus metrics
- FastMCP response size limiting via `ResponseLimitingMiddleware`

Auth policy:
- local `stdio` is treated as trusted
- remote HTTP is protected when `MCP_REQUIRE_AUTH=true`

### 4. MCP metrics

MCP metrics live in `src/app/shared/mcp/metrics.py`.

They reuse the project’s existing Prometheus registry from `src/app/middleware/server_middleware.py`, so `/metrics` includes:
- inbound MCP request counts
- inbound MCP request durations
- MCP tool invocation counts
- MCP tool invocation durations
- outbound MCP client call counts
- outbound MCP client call durations
- upstream MCP server health gauges

### 5. Internal MCP client manager

The internal client layer lives in:
- `src/app/shared/mcp/client.py`
- `src/app/shared/mcp/models.py`

What it does:
- loads approved upstream server configs from settings
- supports HTTP and STDIO upstreams
- supports bearer-token auth for upstream calls
- allowlists approved upstream tools
- adds retries
- adds a simple circuit breaker
- keeps per-server lazy connections
- exposes discovery/status methods
- closes upstream clients during app shutdown

This is not a raw proxy layer. The intent is to keep third-party MCP access explicit and controlled.

## How It Was Implemented

### Server construction

The server is built with `FastMCP(...)` inside `get_mcp_server()`.

Tool registration is done inside `_register_tools(...)` so the public server surface stays centralized and curated.

The registry also:
- paginates list-style results
- truncates oversized payloads
- wraps tool execution with timing and metrics
- normalizes success/error payloads into `MCPToolResponse`

### FastAPI integration

The FastAPI app mounts MCP only after normal routers are attached.

The mounting flow is:
1. Build the normal FastAPI app.
2. Build the MCP ASGI app.
3. Combine FastAPI lifespan and MCP lifespan.
4. Mount MCP at the configured path.

This keeps the project’s existing startup/shutdown behavior intact while enabling FastMCP sessions and HTTP runtime state.

### App-state reuse

The MCP implementation intentionally reuses existing app infrastructure instead of creating parallel resources.

Examples:
- Redis from `app.state` is reused for MCP rate limiting
- the existing lifespan owns resource creation and teardown
- the shared logger is reused for MCP logs
- the existing Prometheus registry is reused for MCP metrics

### Shutdown behavior

The application lifespan closes the internal MCP client manager during shutdown:
- upstream MCP clients do not leak open sessions
- upstream health gauges are reset

## Settings Added

The MCP implementation is controlled from `src/app/config/settings.py`.

Important server settings:
- `MCP_ENABLE_STDIO`
- `MCP_ENABLE_HTTP`
- `MCP_SERVER_NAME`
- `MCP_HTTP_PATH`
- `MCP_HTTP_TRANSPORT`
- `MCP_RUN_TRANSPORT`
- `MCP_HOST`
- `MCP_PORT`
- `MCP_LOG_LEVEL`
- `MCP_REQUIRE_AUTH`
- `MCP_DEFAULT_PAGE_SIZE`
- `MCP_MAX_PAGE_SIZE`
- `MCP_MAX_RESULT_BYTES`
- `MCP_SERVER_ENABLED_TOOLS`
- `MCP_RATE_LIMIT_BURST`
- `MCP_RATE_LIMIT_RATE`
- `MCP_RATE_LIMIT_PERIOD_SECONDS`

Important client settings:
- `MCP_CLIENT_ENABLED`
- `MCP_CLIENT_SERVER_CONFIGS`
- `MCP_CLIENT_DEFAULT_TIMEOUT_SECONDS`
- `MCP_CLIENT_MAX_CONCURRENCY`
- `MCP_CLIENT_RETRY_ATTEMPTS`
- `MCP_CLIENT_CIRCUIT_BREAKER_THRESHOLD`
- `MCP_CLIENT_CIRCUIT_BREAKER_COOLDOWN_SECONDS`

## How To Run It

### Local stdio mode

Use the FastMCP config file:

```bash
uv run fastmcp run fastmcp.json
```

Or run the server entrypoint directly:

```bash
uv run python -m app.shared.mcp.server
```

This mode is intended for local MCP clients and developer workflows.

### Mounted HTTP mode

Run the FastAPI app normally:

```bash
uv run python -m app.server
```

If `MCP_ENABLE_HTTP=true`, the MCP endpoint will be mounted at:

```text
{BACKEND_URL}{MCP_HTTP_PATH}
```

Example:

```text
http://localhost:5000/mcp
```

## Upstream MCP Client Configuration

Approved upstream servers are configured through `MCP_CLIENT_SERVER_CONFIGS`.

It expects a JSON array of objects. Example:

```json
[
  {
    "name": "docs",
    "enabled": true,
    "description": "Internal docs MCP",
    "transport": "http",
    "url": "https://mcp.example.com/mcp",
    "auth_mode": "bearer",
    "bearer_token": "secret-token",
    "namespace": "docs",
    "enabled_tools": ["search", "fetch"],
    "timeout_seconds": 10,
    "retry_attempts": 2,
    "circuit_breaker_threshold": 3,
    "circuit_breaker_cooldown_seconds": 60
  }
]
```

Important:
- upstream servers are opt-in
- upstream tools are opt-in
- arbitrary user-provided MCP URLs are not supported

## Assumptions Made

The current implementation assumes:
- the MCP server should be curated, not generated from all FastAPI routes
- local `stdio` is trusted and primarily for developer tooling
- remote HTTP is a production surface and must be authenticated
- JWT auth should reuse the project’s existing token stack first
- Redis is available in production for MCP rate limiting
- metrics should reuse the existing project registry
- upstream MCP access should be centrally configured and allowlisted
- large MCP responses should be bounded rather than streamed indefinitely in v1

## What Is Intentionally Not Implemented

The current implementation does not try to solve everything.

Deliberate omissions:
- no automatic conversion of the entire REST API into MCP tools
- no arbitrary dynamic upstream server registration
- no external OAuth provider integration for the MCP server yet
- no domain-specific write tools yet
- no rich MCP resources/prompts surface yet
- no automatic search/fetch document backend beyond the current curated catalog/upstream records

This is intentional. The current code establishes a safe production baseline first.

## Recommended Next Extensions

If you extend this MCP layer, do it in this order:

1. Add domain-specific curated tools under `src/app/shared/mcp/registry.py` or split tool registration into small feature modules.
2. Add request metadata propagation into outbound client calls if your upstream servers need user or trace context.
3. Add stronger policy around upstream auth sources if multiple server types are introduced.
4. Add resources/prompts only when a real MCP client needs them.
5. Add richer operational docs for whichever external MCP host/client you actually deploy against.

## Useful File Map

- `src/app/shared/mcp/registry.py`
  Main MCP server factory, tool registration, and HTTP app creation.

- `src/app/shared/mcp/security.py`
  HTTP auth, rate limit, and MCP request observability middleware.

- `src/app/shared/mcp/client.py`
  Internal manager for approved upstream MCP servers.

- `src/app/shared/mcp/models.py`
  MCP config and response models.

- `src/app/shared/mcp/metrics.py`
  MCP-specific metrics registered into the shared Prometheus registry.

- `src/app/shared/mcp/server.py`
  Standalone entrypoint for stdio/CLI use.

- `src/app/main.py`
  FastAPI mounting point for remote HTTP MCP.

- `src/app/lifecycle/lifespan.py`
  Shared resource lifecycle and upstream MCP client shutdown.

## Operational Notes

- If runtime import fails because of settings parsing, fix env issues first. The common one in this repo is invalid boolean parsing for `DEBUG`.
- If you add new MCP tools, keep arguments flat and return shapes predictable.
- If you expose write tools later, split them from read-only tools and apply stricter policy review.
- If you connect to third-party MCP servers, assume they are partial failures and treat retries/circuit breaker state as part of normal operation.

## Chosen Ones

- The most important production decision here is not the middleware. It is refusing to auto-convert your entire API into MCP. That single choice prevents a huge amount of LLM-context bloat, accidental write exposure, and tool-quality collapse.
- The subtle bug to avoid with FastMCP is lifespan ownership. Mounting the HTTP app without combining lifespans often looks fine until session-managed features fail under load or during reconnects.
- The internal MCP client manager matters more than it looks: once teams allow arbitrary upstream MCP passthrough, they usually recreate the same security and observability mistakes that happened with raw plugin ecosystems a decade ago.
