# mcp-server-resources Specification

## Purpose
TBD - created by archiving change mcp-split-enhance. Update Purpose after archive.
## Requirements
### Requirement: Config resource
The system SHALL expose current app configuration as a `@mcp.resource` at `app://config`. Secrets SHALL be redacted before exposure.

#### Scenario: Config resource readable
- **WHEN** an MCP client reads `app://config`
- **THEN** the response contains app name, version, environment, and non-sensitive settings
- **THEN** all `SecretStr` fields are redacted (replaced with `"***"`)
- **THEN** the response uses `application/json` MIME type

### Requirement: Features resource
The system SHALL expose feature flag status as a `@mcp.resource` at `app://features`.

#### Scenario: Features resource readable
- **WHEN** an MCP client reads `app://features`
- **THEN** the response lists all feature flags and their current enabled/disabled state
- **THEN** the response includes feature descriptions

### Requirement: Health resource
The system SHALL expose current health state as a `@mcp.resource` at `app://health`.

#### Scenario: Health resource readable
- **WHEN** an MCP client reads `app://health`
- **THEN** the response contains overall status ("healthy", "degraded", "unhealthy")
- **THEN** the response lists individual dependency status (redis, mongo, db, neo4j, httpx)

### Requirement: Upstream status resource
The system SHALL expose individual upstream MCP server status as a templated `@mcp.resource` at `app://upstreams/{server_name}`.

#### Scenario: Upstream status readable
- **WHEN** an MCP client reads `app://upstreams/docs`
- **THEN** the response contains the status of the "docs" upstream server
- **WHEN** the server_name does not exist
- **THEN** the resource returns a 404-compatible error

### Requirement: Capability catalog resource
The system SHALL expose the full tool/resource/prompt catalog as a `@mcp.resource` at `mcp://catalog`.

#### Scenario: Catalog resource readable
- **WHEN** an MCP client reads `mcp://catalog`
- **THEN** the response lists all registered tools, resources, and prompts
- **THEN** each entry includes name, description, and tags

