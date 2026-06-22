# mcp-telemetry Specification

## Purpose
TBD - created by archiving change mcp-split-enhance. Update Purpose after archive.
## Requirements
### Requirement: FastMCP telemetry hooks wired
The system SHALL wire FastMCP's telemetry hooks (`on_tool_call`, `on_resource_read`, `on_prompt_executed`) to increment existing Prometheus counters from `src/app/middleware/server_middleware.py`.

#### Scenario: Tool call increment counter via hook
- **WHEN** any MCP tool is called
- **THEN** FastMCP's `on_tool_call` hook fires
- **THEN** the `mcp_tool_calls_total` counter is incremented with the correct labels (tool_name, status)
- **THEN** the `mcp_tool_call_duration_seconds` histogram is observed

#### Scenario: Resource read observed
- **WHEN** any MCP resource is read
- **THEN** FastMCP's `on_resource_read` hook fires
- **THEN** a resource-read counter is observed (reuses existing counters or adds new ones)

#### Scenario: Hook covers all tool calls
- **WHEN** a tool is called via CodeMode
- **THEN** the telemetry hook fires (not just calls through the curated tool path)
- **WHEN** a tool is called on a mounted sub-server
- **THEN** the telemetry hook fires

### Requirement: Manual observe_* functions replaced
The manual `observe_mcp_tool_invocation()`, `observe_mcp_client_call()`, and `observe_mcp_http_request()` functions SHALL be migrated to use FastMCP telemetry hooks where possible. Direct call sites SHALL be removed.

#### Scenario: _timed_tool uses hooks
- **WHEN** `_timed_tool()` executes a tool
- **THEN** the metric observation is handled by the telemetry hook, not the manual function
- **THEN** the manual `observe_mcp_tool_invocation()` call inside `_timed_tool()` is removed

### Requirement: Upstream health metric retained
The `mcp_upstream_server_health` gauge SHALL remain a manually set metric since it is set outside MCP tool call context (set by ping results and shutdown).

#### Scenario: Health gauge still works
- **WHEN** an upstream server passes a ping
- **THEN** `set_mcp_upstream_health("server_name", True)` sets the gauge to 1
- **WHEN** an upstream server fails
- **THEN** `set_mcp_upstream_health("server_name", False)` sets the gauge to 0

