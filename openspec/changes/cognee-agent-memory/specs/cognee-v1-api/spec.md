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

### Requirement: Memory vectors are persisted in a durable store, preferring the application's managed database

Agent-memory vectors and their metadata SHALL be persisted in a store whose contents **survive replacement of the
process and of the container that produced them**. The memory library's own default store SHALL NOT be accepted, and
no store whose data is lost on process or container replacement SHALL be configured. The application's managed
relational database is the required target; a file-backed store on durable storage is permitted only under the
conditions in the fallback scenario below.

#### Scenario: Memory vectors survive process replacement

- **WHEN** a memory write completes and the process is then replaced
- **THEN** a subsequent recall from a new process SHALL still find the written memory

#### Scenario: The memory library's default store is never accepted

- **WHEN** the memory subsystem is configured at startup
- **THEN** its vector store SHALL have been configured explicitly rather than left to the library default
- **AND** no store resolving to the producing process's own ephemeral filesystem SHALL be configured
- **AND** if no durable store can be configured, startup SHALL report the subsystem as degraded rather than
  silently accepting the default

#### Scenario: A durable file-backed store is permitted only for memory recall

- **WHEN** the managed relational database cannot host the memory vector store and a file-backed store on durable
  storage is configured instead
- **THEN** that store SHALL serve agent-memory recall only
- **AND** it SHALL NOT serve document retrieval
- **AND** the health surface SHALL report that the memory subsystem is running on the fallback store rather than
  reporting it as fully configured

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

### Requirement: The memory subsystem is configured against the application's own database

The memory subsystem's relational store SHALL be configured from the application's single settings source using the
**discrete connection fields its own configuration surface accepts** — provider, host, port, username, password and
database name. Those fields SHALL resolve to the **same database instance the application's own engine connects
to**; a configuration that can point the memory subsystem at a different instance than the application SHALL NOT be
accepted. No connection field SHALL be satisfied by a placeholder default. Transport-security parameters SHALL be
supplied in the form the memory subsystem's own driver accepts, never appended to a value that driver does not
parse. The subsystem's reported configuration state MUST NOT carry a connection URL that no component consumes as
configuration, because such a value is indistinguishable from configuration to its readers.

#### Scenario: Memory store and application resolve to one database instance

- **WHEN** the memory subsystem is configured at startup
- **THEN** its resolved host, port and database name SHALL equal those the application's own database engine
  resolves for the same run
- **AND** startup SHALL fail rather than configure the memory subsystem against a different instance

#### Scenario: No connection field falls back to a placeholder default

- **WHEN** the memory subsystem resolves its connection fields
- **THEN** each field SHALL come from the application's configured settings
- **AND** a field still holding its placeholder default SHALL be treated as a configuration error rather than as a
  usable value

#### Scenario: Memory store connection succeeds on first use

- **WHEN** the memory subsystem performs its first database operation
- **THEN** the connection SHALL authenticate successfully using the credential drawn from the application's secret
  settings
- **AND** no credential SHALL be supplied by an ambient environment side effect
- **AND** transport security SHALL be negotiated through the driver's own connection arguments

#### Scenario: An unusable connection is reported, not swallowed

- **WHEN** the memory subsystem's database connection cannot authenticate
- **THEN** the memory health check SHALL report a failure naming the store
- **AND** the application SHALL NOT report the memory subsystem as healthy
- **AND** no reported configuration value SHALL expose a credential or a credential-less connection URL

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

#### Scenario: Store final report

- **WHEN** `store_final_report()` is called with an approved report and its conversation identity
- **THEN** `cognee.remember(report_json, dataset_name=dataset_name)` is called

#### Scenario: Store relationships

- **WHEN** relationship data is produced for a document
- **THEN** it SHALL NOT reach `cognee.remember()` — the summary is written to the document knowledge graph instead

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

#### Scenario: Process report after store

- **WHEN** a report is stored on the request path
- **THEN** `cognee.improve(dataset=dataset_name)` is NOT called by that write

#### Scenario: Process relationships after store

- **WHEN** relationship data is produced for a document
- **THEN** no enrichment pass runs for it in agent memory, because nothing was stored there

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

#### Scenario: Search episodic memory

- **WHEN** the agent-memory service is called with a query and partition identity
- **THEN** `cognee.recall(query_text=query, datasets=[partition])` is called

#### Scenario: Search returns results as dicts

- **WHEN** `cognee.recall()` returns a list of results
- **THEN** each result is converted to a dict and returned as a list

#### Scenario: Search handles failures gracefully

- **WHEN** `cognee.recall()` raises an exception
- **THEN** an empty list SHALL be returned and the error SHALL be logged
- **AND** the caller's run SHALL continue

#### Scenario: Search returns results as serialisable mappings

- **WHEN** `cognee.recall()` returns a list of results
- **THEN** each result is converted to a fully serialisable mapping and returned as a list

#### Scenario: Recall results are fully serialisable and retain their origin

- **WHEN** `cognee.recall()` returns results
- **THEN** each result SHALL be converted to a fully serialisable mapping with no nested unserialised objects
- **AND** each result SHALL retain the field identifying whether it came from the conversation cache or from the
  permanent memory graph

#### Scenario: Recall handles failures gracefully

- **WHEN** `cognee.recall()` raises an exception
- **THEN** an empty list SHALL be returned and the error SHALL be logged
- **AND** the caller's run SHALL continue

### Requirement: No type ignore suppressions

The system SHALL NOT use `# type: ignore` comments on the agent-memory call surface — the module or modules that
call `cognee.remember()`, `cognee.improve()` and `cognee.recall()` — whatever their path. Retiring or relocating that
call surface SHALL NOT be a way to satisfy this requirement: the prohibition follows the calls, not the file.

#### Scenario: Type checker passes

- **WHEN** the project's type checker is run over the module or modules that hold the agent-memory call surface
- **THEN** no type errors SHALL be reported on `cognee.remember()`, `cognee.improve()` or `cognee.recall()` calls
- **AND** no `# type: ignore` comment SHALL appear on any line of that call surface
