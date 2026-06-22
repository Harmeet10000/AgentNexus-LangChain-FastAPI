## ADDED Requirements

### Requirement: FastMCP Context in tools
Tools SHALL use FastMCP's `Context` parameter for dependency injection, including logging, progress reporting, and session state.

#### Scenario: Context injected into tool
- **WHEN** a tool function declares a `ctx: Context` parameter
- **THEN** FastMCP injects the current request context automatically
- **THEN** the tool can call `ctx.info()`, `ctx.warning()`, `ctx.error()` for structured logging
- **THEN** the tool can call `ctx.report_progress(progress, total)` for progress updates

#### Scenario: Session state available
- **WHEN** a tool reads `ctx.state`
- **THEN** it accesses a dict scoped to the current MCP session
- **THEN** state persists across tool calls within the same session

### Requirement: Dependency injection via Depends
Tools SHALL use FastMCP's `Depends` for composable dependency injection, replacing manual parameter passing where appropriate.

#### Scenario: Depends injects settings
- **WHEN** a tool declares `settings: AppSettings = Depends(get_settings)`
- **THEN** FastMCP resolves the dependency and injects the settings object
- **THEN** the tool can access settings without calling `get_settings()` directly

### Requirement: Context-based logging
Tool-specific structured logging SHALL use `ctx.info()` / `ctx.warning()` / `ctx.error()` instead of the module-level `logger` singleton where context is available. The module-level logger SHALL remain as fallback for non-request contexts.

#### Scenario: Context logging includes request metadata
- **WHEN** a tool calls `ctx.info("tool completed")`
- **THEN** the log line includes the current session ID, tool name, and correlation ID
- **THEN** the log line is emitted through the project's loguru pipeline
