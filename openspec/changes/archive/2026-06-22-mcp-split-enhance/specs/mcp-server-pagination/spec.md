## ADDED Requirements

### Requirement: FastMCP built-in pagination
The system SHALL use FastMCP's built-in pagination via the `list_max_results` parameter instead of the custom `_paginate()` function.

#### Scenario: FastMCP configured with max page size
- **WHEN** the FastMCP server is created
- **THEN** the constructor receives `list_max_results=settings.MCP_MAX_PAGE_SIZE`
- **THEN** FastMCP handles `tools/list`, `resources/list`, and `prompts/list` pagination with `nextCursor`

### Requirement: Custom _paginate removed
The custom `_paginate()` function in `registry.py` SHALL be removed. All list-style tools SHALL rely on FastMCP's built-in pagination.

#### Scenario: search tool uses native pagination
- **WHEN** the `search` tool returns results
- **THEN** pagination is handled by FastMCP's `list_max_results`, not `_paginate()`
- **THEN** the `limit` and `offset` parameters on the `search` tool are removed (FastMCP handles this at protocol level)

### Requirement: Backward-compatible tool signatures
The `list_upstream_servers` and `search` tools SHALL keep their `limit`/`offset` parameters for backward compatibility with existing callers, but the implementation SHALL delegate to FastMCP's pagination internally.

#### Scenario: Legacy parameters still accepted
- **WHEN** `list_upstream_servers(limit=5, offset=10)` is called
- **THEN** the parameters are accepted and used as before
- **THEN** the implementation delegates to FastMCP pagination under the hood
