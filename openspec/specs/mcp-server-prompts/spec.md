# mcp-server-prompts Specification

## Purpose
TBD - created by archiving change mcp-split-enhance. Update Purpose after archive.
## Requirements
### Requirement: explain-system prompt
The system SHALL expose a `@mcp.prompt` named `explain-system` that describes the application architecture and available MCP surface.

#### Scenario: Prompt returns structured explanation
- **WHEN** an MCP client requests the `explain-system` prompt
- **THEN** the response describes the app name, version, and environment
- **THEN** the response lists available tools, resources, and prompts
- **THEN** the response describes the upstream server configuration

### Requirement: diagnose-issue prompt
The system SHALL expose a `@mcp.prompt` named `diagnose-issue` that templates an incident diagnosis workflow using available MCP tools.

#### Scenario: Prompt returns diagnosis template
- **WHEN** an MCP client requests the `diagnose-issue` prompt
- **THEN** the response includes a structured workflow: health check, upstream checks, catalog search
- **THEN** each step references the specific MCP tool or resource to use

### Requirement: database-query prompt
The system SHALL expose a `@mcp.prompt` named `database-query` that provides a safe SQL query pattern with guardrails. **Note:** This prompt is a template only — execution requires a database tool to be added in a future change.

#### Scenario: Prompt returns SQL workflow
- **WHEN** an MCP client requests the `database-query` prompt
- **THEN** the response includes safe-SQL guidelines (read-only, no DDL, LIMIT clauses)
- **THEN** the response suggests available resources for schema discovery

### Requirement: deploy-check prompt
The system SHALL expose a `@mcp.prompt` named `deploy-check` that provides a pre-deployment checklist.

#### Scenario: Prompt returns deployment checklist
- **WHEN** an MCP client requests the `deploy-check` prompt
- **THEN** the response includes steps: health check, upstream pings, feature flag config review
- **THEN** each step references specific MCP tools/resources

### Requirement: health-report prompt
The system SHALL expose a `@mcp.prompt` named `health-report` that summarizes system health.

#### Scenario: Prompt returns health summary workflow
- **WHEN** an MCP client requests the `health-report` prompt
- **THEN** the response includes steps to read `app://health` and `app://upstreams/*` and interpret results

