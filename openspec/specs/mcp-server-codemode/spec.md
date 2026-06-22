# mcp-server-codemode Specification

## Purpose
TBD - created by archiving change mcp-split-enhance. Update Purpose after archive.
## Requirements
### Requirement: CodeMode enabled
The system SHALL enable FastMCP CodeMode on the server, providing a 3-stage tool discovery pipeline: search, schema inspection, and sandboxed execution.

#### Scenario: CodeMode tools available
- **WHEN** an MCP client lists tools on a CodeMode-enabled server
- **THEN** the tool list includes `search_tools`, `get_tool_schema`, and (if sandbox enabled) a code execution tool
- **THEN** regular curated tools (health_check, readiness_check, etc.) are also listed

#### Scenario: search_tools returns matches
- **WHEN** an LLM calls `search_tools(query="health")`
- **THEN** the result includes `health_check` and related tools
- **THEN** the result includes tool descriptions

#### Scenario: get_tool_schema returns schema
- **WHEN** an LLM calls `get_tool_schema(tool_name="readiness_check")`
- **THEN** the result includes parameter names, types, and descriptions
- **THEN** the result includes the return type

### Requirement: CodeMode configuration
CodeMode SHALL be configured with sandboxed execution enabled, restricted filesystem access, and no network access in the sandbox.

#### Scenario: Sandbox restrictions applied
- **WHEN** a CodeMode code execution is triggered
- **THEN** the sandbox subprocess has no network access
- **THEN** the sandbox subprocess has read-only access to a restricted temp directory
- **THEN** execution is bounded by a timeout (default 30s)

### Requirement: CodeMode coexists with curated tools
Curated tools SHALL remain available alongside CodeMode tools. The curated tool set provides the stable API; CodeMode provides dynamic discovery.

#### Scenario: Both sets available
- **WHEN** an MCP client lists tools
- **THEN** both curated tools and CodeMode tools appear in the list
- **THEN** curated tool descriptions take precedence over any CodeMode-generated descriptions

