## ADDED Requirements

### Requirement: Role tool bindings are explicit and scope-safe

The authoritative registry MUST expose only the documented role bindings: risk gets evidence search/read, obligation
chain, typed knowledge-graph query, and handoff; compliance gets evidence search/read, statute retrieval, and handoff;
the orchestrator gets ingestion state. Bound tools MUST be native `@tool` functions with uniform argument schemas;
MCP exposure is out of scope. No bound tool may accept model-supplied identity or scope as authority.

#### Scenario: Orchestrator cannot read legal evidence directly
- **WHEN** the orchestrator is constructed
- **THEN** it MUST receive ingestion-state and delegation tools but no direct evidence read tool

#### Scenario: Tool binding cannot grant identity
- **WHEN** a model supplies user, tenant, thread, step, or document authority arguments
- **THEN** the tool MUST ignore or reject those arguments and use runtime context

### Requirement: Read-only and mutating operations have distinct gates

Read-only evidence and knowledge tools MUST be runnable without an approval interrupt once the server-derived context
authorizes the operation. Any mutating operation MUST be unavailable until an authorized human approval has completed,
and approval/rejection/override MUST remain server-role-checked on every resume. Upload, resume, approval, and
re-ingestion MUST not be model-callable tools; uploads enter through HTTP and the agent receives document identifiers.

#### Scenario: Read-only evidence search runs without approval
- **WHEN** an authorized risk or compliance node invokes evidence search or read
- **THEN** the tool MUST execute using runtime scope without requiring a human approval step

#### Scenario: Mutating operation is approval-gated
- **WHEN** a model attempts to invoke a mutating operation before authorized human approval
- **THEN** the registry MUST not expose or execute that operation and MUST leave persisted state unchanged

#### Scenario: Tool transport remains native
- **WHEN** the Saul tool registry is constructed
- **THEN** every bound tool MUST be a native `@tool` function and no MCP upload, resume, approval, or re-ingestion tool MUST be registered
