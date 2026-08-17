## ADDED Requirements

### Requirement: Memory embeddings match the document embedding dimensionality

Agent-memory embeddings SHALL be produced at the same vector dimensionality as the application's document
embeddings, using the same embedding provider family. The dimensionality SHALL be derived from the application's
single configured embedding dimension and MUST NOT be left to a third-party default.

#### Scenario: Memory embedding dimensionality equals document embedding dimensionality

- **WHEN** the memory subsystem is configured at startup
- **THEN** its embedding dimensionality SHALL equal the application's configured embedding dimension
- **AND** startup SHALL fail rather than proceed if the two differ

#### Scenario: The third-party embedding default is never used

- **WHEN** the memory subsystem resolves its embedding model
- **THEN** the resolved model SHALL be the application's configured embedding model
- **AND** it SHALL NOT be a provider default that requires credentials the application has not configured

### Requirement: Memory vectors are persisted in the application's managed database

Agent-memory vectors and their metadata SHALL be persisted in the application's managed relational database. They
MUST NOT be written to the local filesystem of the process that produced them.

#### Scenario: Memory vectors survive process replacement

- **WHEN** a memory write completes and the process is then replaced
- **THEN** a subsequent recall from a new process SHALL still find the written memory

#### Scenario: No memory data is written to local files

- **WHEN** the memory subsystem is configured at startup
- **THEN** no memory vector store SHALL be configured against a local filesystem path
- **AND** if no managed vector store is available, startup SHALL report the subsystem as degraded rather than
  silently falling back to local files

### Requirement: Memory multi-user access control state is explicit

The memory subsystem's multi-user access-control setting SHALL be set explicitly at startup, before the first
memory configuration call. It MUST NOT be left unset.

#### Scenario: Access control is explicitly disabled

- **WHEN** the application starts
- **THEN** the memory subsystem's access-control setting SHALL hold an explicit value
- **AND** the first memory write SHALL NOT raise an environment error caused by an unset setting

#### Scenario: Tenant isolation is enforced above the memory library

- **WHEN** access control is disabled
- **THEN** tenant isolation SHALL be enforced by the application through the memory partition identity
- **AND** the partition identity SHALL be produced by a single validated construction path

### Requirement: The memory subsystem receives an authenticated database connection

The memory subsystem SHALL receive a database connection string that is usable as given — correct scheme,
credentials present, and no transport parameters the driver rejects. It MUST NOT read a raw configuration value
that bypasses the application's single connection-string accessor.

#### Scenario: Memory store connection succeeds on first use

- **WHEN** the memory subsystem performs its first database operation
- **THEN** the connection SHALL authenticate successfully
- **AND** no credential SHALL be supplied by an ambient environment side effect

#### Scenario: An unusable connection is reported, not swallowed

- **WHEN** the memory subsystem's database connection cannot authenticate
- **THEN** the memory health check SHALL report a failure naming the store
- **AND** the application SHALL NOT report the memory subsystem as healthy

## MODIFIED Requirements

### Requirement: Store content via remember

The system SHALL use `cognee.remember()` to store content in agent memory, replacing the deprecated `cognee.add()`.
Every such write SHALL be scoped to a conversation identity, because typed memory entries cannot be written without
one, and SHALL be made in conversation-scoped mode so that no full graph rebuild or enrichment pass is performed on
the request path.

#### Scenario: Approved final report is stored in conversation scope

- **WHEN** an approved final report is stored in agent memory
- **THEN** `cognee.remember()` SHALL be called with the run's conversation identity
- **AND** it SHALL be called with self-improvement disabled, so no detached background enrichment is started

#### Scenario: A write without a conversation identity is rejected

- **WHEN** a caller attempts to store a typed memory entry with no conversation identity
- **THEN** the write SHALL be rejected before it reaches the memory library
- **AND** the rejection SHALL be reported as a caller error, not as a memory-store failure

#### Scenario: Relationship summaries are no longer stored in agent memory

- **WHEN** relationship data is produced for a document
- **THEN** it SHALL be written to the document knowledge graph
- **AND** it SHALL NOT be written into agent memory as text

### Requirement: Process content via improve

The system SHALL use `cognee.improve()` to consolidate conversation-scoped memory into the permanent memory graph,
replacing the deprecated `cognee.cognify()`. Consolidation SHALL be invoked **only** by the scheduled consolidation
job and SHALL NOT be invoked after an individual write, because the write API already performs its own enrichment
when it is not conversation-scoped.

#### Scenario: A write does not trigger consolidation

- **WHEN** content is successfully stored in agent memory on the request path
- **THEN** no consolidation call SHALL follow that write

#### Scenario: Consolidation is invoked only on a schedule

- **WHEN** the scheduled consolidation job runs
- **THEN** `cognee.improve()` SHALL be called once for the dataset with the conversation identities to consolidate
- **AND** no request-path code path SHALL be able to invoke consolidation

### Requirement: Query memory via recall

The system SHALL use `cognee.recall()` to query stored memories, replacing the deprecated `cognee.search()`. The
query type SHALL be auto-routed by Cognee (default `auto_route=True`) — no explicit `SearchType` enum is required.

#### Scenario: Recall is scoped to the caller's memory partition

- **WHEN** agent memory is queried during a run
- **THEN** `cognee.recall()` SHALL be called with the caller's memory partition and conversation identity
- **AND** results from another tenant's partition SHALL NOT be returned

#### Scenario: Recall results are fully serialisable and retain their origin

- **WHEN** `cognee.recall()` returns results
- **THEN** each result SHALL be converted to a fully serialisable mapping with no nested unserialised objects
- **AND** each result SHALL retain the field identifying whether it came from the conversation cache or from the
  permanent memory graph

#### Scenario: Recall handles failures gracefully

- **WHEN** `cognee.recall()` raises an exception
- **THEN** an empty list SHALL be returned and the error SHALL be logged
- **AND** the caller's run SHALL continue
