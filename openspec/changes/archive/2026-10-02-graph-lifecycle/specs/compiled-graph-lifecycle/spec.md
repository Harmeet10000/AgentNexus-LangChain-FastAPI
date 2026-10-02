## Purpose

Compiled graphs SHALL be built once per process rather than per invocation or at import, job-scoped
collaborators SHALL be supplied per invocation rather than captured, provisioning failure SHALL
degrade rather than abort startup, and the process that constructs a resource SHALL release it.

## ADDED Requirements

### Requirement: Compiled graphs are process-scoped, not per-invocation

A graph SHALL be compiled at most once per process, and never as a side effect of importing a module.

#### Scenario: Two jobs in one worker process compile once

- **WHEN** two document-ingestion jobs execute in the same worker process
- **THEN** the graph SHALL be compiled at most once

#### Scenario: Startup compiles each provisioned graph once

- **WHEN** the serving process completes startup
- **THEN** each provisioned graph SHALL have been compiled exactly once

#### Scenario: Import compiles nothing

- **WHEN** a graph module is imported
- **THEN** no graph SHALL be compiled as a side effect of that import

### Requirement: Job-scoped collaborators are supplied per invocation

A compiled graph SHALL NOT capture request-scoped or job-scoped state, and SHALL receive it at
invocation time instead.

#### Scenario: The repository arrives through the invocation configuration

- **WHEN** a document-ingestion run is invoked
- **THEN** the job-scoped repository SHALL be supplied through the invocation configuration

#### Scenario: Compilation captures no session

- **WHEN** a graph is compiled
- **THEN** it SHALL NOT capture a database session or repository, and graph state SHALL carry neither

#### Scenario: A missing collaborator is a typed failure

- **WHEN** an invocation omits the required job-scoped collaborator
- **THEN** a typed exception naming the missing collaborator SHALL be raised, and not a key or
  attribute error

### Requirement: Graph provisioning degrades rather than failing startup

A graph that cannot be built SHALL leave the process running and its capability observably absent.

#### Scenario: A build failure does not abort startup

- **WHEN** a graph fails to build during startup
- **THEN** startup SHALL continue and that capability's state attribute SHALL be absent or none

#### Scenario: Reading an unprovisioned graph names the capability

- **WHEN** a dependency reads an unprovisioned graph
- **THEN** it SHALL raise a typed service-unavailable naming the missing capability, rather than
  producing a server error

#### Scenario: Provisioning reuses the shared degrade machinery

- **WHEN** a graph is registered for provisioning
- **THEN** it SHALL be registered through the existing startup-policy mechanism rather than through a
  bespoke exception block

### Requirement: Agent Saul is provisioned in the serving process

The Agent Saul graph and the checkpointer it requires SHALL be available on a healthy serving process.

#### Scenario: Both are readable from process state after startup

- **WHEN** the serving process has started successfully
- **THEN** the Agent Saul graph and the checkpointer SHALL both be readable from application state

#### Scenario: A healthy process does not report the capability unavailable

- **WHEN** a request reaches the Agent Saul dependency bundle on a healthy process
- **THEN** it SHALL NOT report the capability as unavailable

#### Scenario: The checkpointer is constructed before the graph that requires it

- **WHEN** the Agent Saul graph is constructed
- **THEN** its required checkpointer SHALL already exist, because that parameter admits no absent value

### Requirement: Memory behaviour is supplied, not redefined

Agent memory SHALL be provided to the graph as a collaborator, and its behaviour SHALL remain governed
by the existing memory capability.

#### Scenario: The memory service is constructed once and passed through

- **WHEN** the serving process provisions Agent Saul
- **THEN** the memory service SHALL be constructed in that provisioning path and supplied to the
  graph, rather than constructed a second time elsewhere

#### Scenario: No memory semantics are redefined here

- **WHEN** this capability is read
- **THEN** it SHALL state nothing about how memory is recalled, written, or partitioned

### Requirement: The constructing process releases what it constructs

Resources hoisted to process scope SHALL be released when that process shuts down, and the release
SHALL be observable.

#### Scenario: The serving process closes its checkpointer pool

- **WHEN** the serving process shuts down after provisioning a checkpointer
- **THEN** its connection pool SHALL be closed and the outcome reported

#### Scenario: The worker process releases its per-process resources

- **WHEN** a worker process shuts down after constructing per-process resources
- **THEN** each SHALL be released exactly once, and no release SHALL be attempted for a resource that
  was never constructed

#### Scenario: No credential is emitted during setup or teardown

- **WHEN** a checkpointer setup or teardown fails
- **THEN** no credential SHALL appear in any emitted log line
