## Purpose

Defines the behaviour of the legal agent's long-term memory: which artifacts of a run are remembered, the partition
they are remembered under, when memory is read back into a run, how memory failures are contained so that they never
fail a completed legal analysis, when accumulated memory is consolidated into the permanent memory graph, and how the
memory subsystem's own health is reported. This capability covers the agent-run axis only — the document and entity
axis, including all bitemporal validity of extracted obligations, belongs to the document knowledge graph.

## ADDED Requirements

### Requirement: Only human-approved final reports are persisted to agent memory

The agent SHALL persist a final report to agent memory only after human approval has been granted. Reports that did
not pass the approval gate SHALL NOT be persisted at any confidence or trust level.

#### Scenario: Approved final report is persisted

- **WHEN** the agent run reaches memory persistence with human approval granted
- **THEN** the final report SHALL be written to agent memory before the run completes

#### Scenario: Unapproved run persists nothing

- **WHEN** the agent run reaches memory persistence and human approval has not been granted
- **THEN** no final report SHALL be written to agent memory
- **AND** the run SHALL still complete successfully

### Requirement: Final reports are written to agent memory only

The final report SHALL have exactly one memory owner. It SHALL NOT be written to the document knowledge graph, and no
final-report write path into that graph SHALL be invoked.

#### Scenario: Final report storage has a single owner

- **WHEN** an approved final report is persisted
- **THEN** it SHALL be written to agent memory
- **AND** no document-knowledge-graph write SHALL be performed for that report

#### Scenario: The system of record is unchanged

- **WHEN** an approved final report is persisted to agent memory
- **THEN** the relational record of that report SHALL remain the authoritative copy
- **AND** the agent-memory copy SHALL be treated as a recall-optimised duplicate, never as the record

#### Scenario: Document-axis facts still reach the knowledge graph

- **WHEN** an approved report concerns obligations extracted from a document
- **THEN** those obligations SHALL reach the document knowledge graph through the document extraction path
- **AND** SHALL NOT reach it by re-ingesting the report's prose

### Requirement: Agent memory is partitioned by tenant and conversation

Every agent-memory write and read SHALL be partitioned by the owning tenant and by the conversation the run belongs
to. A read SHALL NOT be able to return entries from another tenant's partition. The partition identity SHALL be
produced by a single validated construction path, not assembled ad hoc at each call site.

#### Scenario: Two tenants never share a partition

- **WHEN** two different tenants each write agent memory
- **THEN** their partition identities SHALL differ
- **AND** neither tenant's recall SHALL return the other's entries

#### Scenario: A write with no conversation identity is refused

- **WHEN** a caller attempts an agent-memory write without a conversation identity
- **THEN** the write SHALL be refused
- **AND** the refusal SHALL be surfaced as a caller error rather than silently writing to a default partition

#### Scenario: Partition identity has one construction path

- **WHEN** any component needs a memory partition identity
- **THEN** it SHALL obtain that identity from the single validated construction path
- **AND** an identity that fails validation SHALL NOT be used for a write or a read

### Requirement: Request-path memory writes do not rebuild the memory graph

An agent-memory write performed while a request or agent run is in flight SHALL NOT trigger a full rebuild of the
permanent memory graph, and SHALL NOT trigger an enrichment pass, whether synchronously or as detached background
work.

#### Scenario: A report write completes without a graph rebuild

- **WHEN** an approved final report is written to agent memory during a run
- **THEN** the write SHALL complete without rebuilding the permanent memory graph

#### Scenario: No detached enrichment is started by a write

- **WHEN** an agent-memory write completes on the request path
- **THEN** no background enrichment task SHALL be started whose lifetime is not owned by the application

### Requirement: Consolidation into the permanent memory graph runs on a schedule

Consolidation of accumulated conversation-scoped memory into the permanent memory graph SHALL run as a scheduled
background job, separate from any request or agent run. The job SHALL be registered with the application's task
system and SHALL appear in its schedule.

#### Scenario: Consolidation is scheduled, never inline

- **WHEN** the application is running
- **THEN** consolidation SHALL be reachable only through the scheduled job
- **AND** no request-path code SHALL be able to invoke it

#### Scenario: Consolidation reports what it consolidated

- **WHEN** the scheduled consolidation job completes
- **THEN** it SHALL report the number of conversations consolidated and the resulting memory size
- **AND** that report SHALL be observable without reading the memory stores directly

#### Scenario: Consolidation refuses to run when its graph preconditions are absent

- **WHEN** the scheduled consolidation job starts and the required graph procedures are not available on the graph
  database
- **THEN** the job SHALL abort with a named precondition failure
- **AND** it SHALL NOT report success, because the underlying rebuild fails without raising

### Requirement: Agent memory is prefetched after clarification and before deeper reasoning

The agent SHALL prefetch memory context after the clarification step and before deeper reasoning nodes execute.
Prefetch SHALL query agent memory first and MAY add a bounded supplement from the document knowledge graph for
grounding.

#### Scenario: Prefetch runs after clarification

- **WHEN** the clarification step produces the clarified intent
- **THEN** memory context SHALL be prefetched before any deeper reasoning node is invoked

#### Scenario: Agent memory is the primary recall source

- **WHEN** the prefetch step runs
- **THEN** agent memory SHALL be queried first

#### Scenario: The knowledge-graph supplement stays bounded

- **WHEN** the prefetch step adds context from the document knowledge graph
- **THEN** that supplement SHALL be limited to matter and document grounding context
- **AND** SHALL be bounded in size rather than unbounded by the query

#### Scenario: Deep retrieval is not performed for every task

- **WHEN** the prefetch step runs for a task that is neither risk analysis nor compliance
- **THEN** no deeper memory retrieval SHALL be performed

### Requirement: Agent memory failures never fail the run

Agent-memory read and write failures SHALL fail open. A memory failure SHALL NOT abort, roll back, or mark as failed
an agent run whose analysis has otherwise completed.

#### Scenario: A memory read failure degrades to available context

- **WHEN** agent-memory retrieval fails during prefetch
- **THEN** the run SHALL continue using current-run context and any supplement already obtained

#### Scenario: A memory write failure is recorded, not raised

- **WHEN** an agent-memory write fails during memory persistence
- **THEN** the run SHALL complete
- **AND** a named memory-write error SHALL be recorded on the run's errors
- **AND** no exception SHALL propagate out of the memory persistence step

### Requirement: The agent memory subsystem reports its own health

The application's health surfaces SHALL report the state of the agent-memory subsystem, distinguishing *not
configured* from *unreachable* from *healthy*. The graph-procedure precondition required for consolidation SHALL be
reported as a named sub-field of that check.

#### Scenario: Unconfigured memory reports degraded

- **WHEN** the memory subsystem's configuration is absent
- **THEN** the health check SHALL report the memory subsystem as degraded
- **AND** it SHALL NOT report it as healthy and SHALL NOT fail the request

#### Scenario: Unreachable memory stores report failure

- **WHEN** the memory subsystem is configured but its stores cannot be reached
- **THEN** the health check SHALL report a failure naming the memory subsystem

#### Scenario: The graph-procedure precondition is reported

- **WHEN** the memory health check runs
- **THEN** it SHALL report whether the graph database exposes the procedures required for consolidation
- **AND** an absent precondition SHALL be reported as a named sub-field rather than failing the whole check

#### Scenario: Both health surfaces agree

- **WHEN** either of the application's health surfaces is queried
- **THEN** each SHALL include the agent-memory subsystem
- **AND** both SHALL report the same state for it
