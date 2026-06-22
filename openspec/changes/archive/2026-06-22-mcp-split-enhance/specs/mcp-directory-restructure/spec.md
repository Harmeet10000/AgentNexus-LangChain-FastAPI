## ADDED Requirements

### Requirement: Directory restructure
The system SHALL reorganize `src/app/shared/mcp/` into `src/app/mcp/{server,client,common}/` with clear separation of server, client, and common concerns. All existing functionality MUST be preserved.

#### Scenario: Files moved to correct locations
- **WHEN** the restructure is applied
- **THEN** `src/app/mcp/server/factory.py` contains the FastMCP server factory
- **THEN** `src/app/mcp/server/tools.py` contains tool definitions
- **THEN** `src/app/mcp/server/middleware.py` contains ASGI middleware
- **THEN** `src/app/mcp/server/http.py` contains the HTTP app builder
- **THEN** `src/app/mcp/client/manager.py` contains MCPClientManager
- **THEN** `src/app/mcp/client/auth.py` contains token exchange and storage
- **THEN** `src/app/mcp/client/settings.py` contains client config loading
- **THEN** `src/app/mcp/common/models.py` contains shared DTOs
- **THEN** `src/app/mcp/common/settings.py` contains MCP-specific settings
- **THEN** `src/app/mcp/testing.py` contains test helpers
- **THEN** `cli.py` contains the standalone CLI entrypoint

### Requirement: Import path migration
All internal and external imports from `app.shared.mcp` SHALL be updated to `app.mcp.*`.

#### Scenario: main.py updated
- **WHEN** the restructure is applied
- **THEN** `src/app/main.py` imports from `app.mcp` instead of `app.shared.mcp`
- **THEN** `get_mcp_http_app` and `parse_mcp_http_transport` resolve correctly

#### Scenario: connections/mcp.py updated
- **WHEN** the restructure is applied
- **THEN** `src/app/connections/mcp.py` imports from `app.mcp` instead of `app.shared.mcp`
- **THEN** `get_mcp_client_manager` resolves correctly

#### Scenario: lifespan.py updated
- **WHEN** the restructure is applied
- **THEN** `src/app/lifecycle/lifespan.py` imports from `app.mcp` if MCP client manager wiring is uncommented
- **THEN** MCP client manager is initialized on startup and closed on shutdown

### Requirement: Compat shim
The system SHALL provide a backward-compat shim at `src/app/shared/mcp/__init__.py` that re-exports all symbols from `app.mcp.*` with a `DeprecationWarning`.

#### Scenario: Deprecation warning issued
- **WHEN** any code imports from `app.shared.mcp`
- **THEN** a `DeprecationWarning` is emitted
- **THEN** the original symbol resolves correctly via the re-export

#### Scenario: All symbols re-exported
- **WHEN** the shim is loaded
- **THEN** all 12 symbols from the original `__init__.py` `__all__` are available

### Requirement: Old module removed
The `src/app/shared/mcp/` directory SHALL be removed after the compat shim window expires.

#### Scenario: Cleanup ticket filed
- **WHEN** the restructure change is marked complete
- **THEN** a CI check or ticket tracks removal of `src/app/shared/mcp/`
- **THEN** a `TODO` comment marks the cleanup target
