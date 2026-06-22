## ADDED Requirements

### Requirement: MCPTestClient helper
The system SHALL provide a `MCPTestClient` class in `src/app/mcp/testing.py` that wraps a FastMCP server for in-process testing without starting a real HTTP or stdio transport.

#### Scenario: TestClient calls tool
- **WHEN** a test creates `MCPTestClient(server)` and calls `await client.call_tool("health_check")`
- **THEN** the tool executes in-process
- **THEN** the result is returned as a dict

#### Scenario: TestClient reads resource
- **WHEN** a test calls `await client.read_resource("app://health")`
- **THEN** the resource handler executes in-process
- **THEN** the result is returned as structured content

#### Scenario: TestClient works as context manager
- **WHEN** `async with MCPTestClient(server) as client:` is used
- **THEN** the client is properly initialized on enter
- **THEN** the client is properly cleaned up on exit

### Requirement: pytest fixtures
The system SHALL provide pytest fixtures for common MCP test patterns.

#### Scenario: mcp_server fixture
- **WHEN** a test requests the `mcp_server` fixture
- **THEN** it receives a FastMCP server instance with a minimal test tool set
- **THEN** the server is configured for in-process use

#### Scenario: mcp_client fixture
- **WHEN** a test requests the `mcp_client` fixture
- **THEN** it receives a connected `MCPTestClient` instance
- **THEN** the client can call tools and read resources

### Requirement: Test tool helper
The system SHALL provide a helper to register test-only tools on a server for unit testing.

#### Scenario: Test tool registered
- **WHEN** a test uses the helper to register a test tool
- **THEN** the tool is available on the server
- **THEN** the tool is not visible in production (separate server instance)

### Requirement: Existing conftest mock updated
The mock in `tests/conftest.py` SHALL be updated to mock `app.mcp` instead of `app.shared.mcp`.

#### Scenario: Conftest mock paths updated
- **WHEN** `tests/conftest.py` loads
- **THEN** `sys.modules["app.mcp"]` is mocked (replacing `app.shared.mcp`)
- **THEN** `sys.modules["app.mcp.server.middleware"]` is mocked (replacing `app.shared.mcp.security`)
- **THEN** existing tests pass without modification
