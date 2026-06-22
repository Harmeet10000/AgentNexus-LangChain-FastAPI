# mcp-server-composition Specification

## Purpose
TBD - created by archiving change mcp-split-enhance. Update Purpose after archive.
## Requirements
### Requirement: Trusted upstream mounting via create_proxy
The system SHALL support mounting trusted upstream MCP servers as sub-servers using FastMCP's `create_proxy()`. Tools from trusted upstreams SHALL be auto-discovered and accessible under a configurable namespace prefix.

#### Scenario: Upstream mounted and tools discovered
- **WHEN** a trusted upstream MCP server is configured with `proxy: true`
- **THEN** `create_proxy()` is called with the upstream URL
- **THEN** the proxy is mounted via `mcp.mount(proxy, namespace="<namespace>")`
- **THEN** all tools from the upstream appear under the `<namespace>_` prefix
- **THEN** tool calls route through FastMCP's native transport, not the custom manager

### Requirement: Tier 2 upstreams retain MCPClientManager
The system SHALL keep the existing `MCPClientManager` for upstream servers requiring per-tool allowlist enforcement. The manager SHALL coexist with `create_proxy()`-mounted upstreams.

#### Scenario: Allowlisted upstream uses custom manager
- **WHEN** an upstream server config has `enabled_tools` set (non-empty)
- **THEN** the server is managed through `MCPClientManager`, not `create_proxy()`
- **THEN** only the listed tools are exposed via the existing `list_upstream_servers` / `call_tool` path

#### Scenario: Mixed deployment works
- **WHEN** both proxy-mounted and manager-managed upstreams are configured
- **THEN** tools from both are available
- **THEN** the `list_upstream_servers` tool reports both types with a `connection_type` field (`proxy` or `managed`)

### Requirement: Proxy upstream circuit breaking
The system SHALL apply timeout and error handling to proxy-mounted upstreams via FastMCP's built-in client configuration.

#### Scenario: Proxy upstream timeout respected
- **WHEN** a proxy-mounted upstream exceeds the configured timeout
- **THEN** the tool call returns an error
- **THEN** the error is logged via the telemetry hooks

