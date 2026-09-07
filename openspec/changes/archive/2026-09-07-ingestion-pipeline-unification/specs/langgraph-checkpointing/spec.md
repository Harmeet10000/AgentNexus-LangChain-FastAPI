## Purpose

Define the persistence and resumability contract for the ingestion pipeline: a checkpointer that is actually
constructible against a real database driver, pool ownership that belongs to whichever process constructs it and a
teardown that can tell "closed a pool" from "there was nothing to close", thread identity tied to the document, and
checkpointed state that carries references rather than document payloads.

**Two boundaries this capability deliberately does not cross.** It provisions the checkpointer; it does **not**
define what a *consumer* does when the deployment deliberately leaves the shared checkpointer unprovisioned — that
read-site fail-closed contract belongs to `agent-runtime-resilience` in the agent-tools change, which owns the
unguarded read of the shared application slot. And it consumes a connection-string accessor; it does **not** define
the accessor set — that belongs to `infrastructure-client-access` in the foundation change. This capability names
the flavour it needs and the properties that flavour must have, and nothing more.

## ADDED Requirements

### Requirement: The pipeline accepts a checkpointer and a thread identity
The pipeline SHALL accept a checkpointer when it is constructed and SHALL be invoked with an explicit thread
identity derived from the document identity. Thread identity SHALL be supplied as invocation configuration, not
as a pipeline state value.

#### Scenario: Pipeline construction accepts a checkpointer
- **WHEN** the pipeline is constructed with a checkpointer
- **THEN** invocations of that pipeline SHALL persist their state at each stage boundary

#### Scenario: Thread identity is derived from the document
- **WHEN** a document's ingestion is invoked
- **THEN** the invocation SHALL supply a thread identity derived from that document's identity

#### Scenario: Construction without a checkpointer is explicit
- **WHEN** the pipeline is constructed without a checkpointer
- **THEN** it SHALL run without persistence and SHALL record that persistence is unavailable

### Requirement: Resuming a thread does not re-run completed stages
When an invocation resumes an existing thread after a failure, the system SHALL resume at the failed stage and
SHALL NOT re-execute stages that already completed for that thread.

#### Scenario: A late-stage failure resumes at the failed stage
- **WHEN** a pipeline invocation fails at a late stage and is re-invoked with the same thread identity
- **THEN** the earlier completed stages SHALL NOT execute again

#### Scenario: A new thread runs every stage
- **WHEN** a pipeline invocation uses a thread identity with no stored checkpoint
- **THEN** every stage SHALL execute from the first

### Requirement: Durability mode is declared explicitly
The system SHALL declare the durability mode under which checkpoints are written, and that mode SHALL be
recoverable from a failure occurring mid-execution rather than only at completion.

#### Scenario: A mid-execution crash is recoverable
- **WHEN** the process terminates while a stage is executing
- **THEN** the last completed stage's checkpoint SHALL be readable and the thread SHALL be resumable from it

#### Scenario: The mode is stated, not implied
- **WHEN** a reader inspects the persistence contract
- **THEN** the durability mode in force SHALL be stated explicitly

### Requirement: Checkpointed state carries references, not document payloads
Checkpointed pipeline state SHALL carry identifiers, references, and small scalars. It SHALL NOT carry raw
document bytes or full document text in any channel, including accumulated per-chunk channels, and the serialised
state for one document SHALL remain within a stated size budget.

#### Scenario: State round-trips through the serialiser
- **WHEN** a populated pipeline state is serialised and deserialised
- **THEN** it SHALL round-trip to an equal value without relying on an arbitrary-object fallback

#### Scenario: Serialised state stays within budget
- **WHEN** a populated pipeline state for a large document is serialised
- **THEN** the serialised payload SHALL be within the stated size budget

#### Scenario: No raw document bytes channel exists
- **WHEN** the pipeline state channels are inspected
- **THEN** none SHALL hold raw document bytes, and the failure channel SHALL hold a plain serialisable record rather than an exception instance

#### Scenario: Accumulated per-chunk items stay small
- **WHEN** per-chunk results accumulate across a fan-out
- **THEN** each accumulated item SHALL carry references and metadata rather than the chunk's full source text

### Requirement: The constructing process owns the checkpointer pool, and teardown distinguishes nothing-to-close from a close
Whichever process constructs the checkpointer SHALL own the connection pool that checkpointer uses and SHALL close
that pool when that process shuts down. In this change that process is the queue worker, not the application serving
HTTP requests: the application's construction of the checkpointer is deliberately disabled, and this change SHALL
NOT enable it. The shutdown path that calls teardown SHALL complete without raising when no checkpointer was ever
constructed, and SHALL report that outcome distinguishably from a successful close rather than returning silently.

#### Scenario: The owning process closes the pool it created
- **WHEN** the process that constructed a checkpointer shuts down
- **THEN** that checkpointer's connection pool SHALL be closed, and the closure SHALL be observable

#### Scenario: Teardown with no checkpointer is reported, not silent
- **WHEN** the shutdown path calls teardown and no checkpointer was ever constructed
- **THEN** teardown SHALL complete without raising and SHALL report that no checkpointer was provisioned, rather than returning with no observable outcome

#### Scenario: Teardown does not silently skip a pool it was given
- **WHEN** teardown receives a value that is not a checkpointer exposing a pool
- **THEN** it SHALL report that it could not close a pool rather than completing as though it had

#### Scenario: The deliberately disabled application construction stays disabled
- **WHEN** the application's startup path is inspected
- **THEN** the checkpointer construction there SHALL remain disabled, and this change SHALL introduce no flag, default, or alternative path that enables it

### Requirement: Checkpointer setup creates its storage before first use
Checkpointer setup SHALL create the checkpoint storage it requires before the first checkpoint is written, and
SHALL yield a usable saver instance rather than an unentered resource manager.

#### Scenario: Setup yields a usable saver
- **WHEN** checkpointer setup completes
- **THEN** the returned value SHALL be a checkpointer able to read and write checkpoints immediately

#### Scenario: Checkpoint storage exists after setup
- **WHEN** setup completes against a reachable database
- **THEN** the checkpoint storage tables SHALL exist

### Requirement: A missing database driver fails loudly at import
When the checkpointer's database driver cannot load, the failure SHALL surface as an import failure. The system
SHALL NOT alias the checkpointer type to a permissive placeholder, SHALL NOT return an absent checkpointer from a
function declared to return one, and SHALL NOT defer the failure to the first consumer.

#### Scenario: Absent driver surfaces at import
- **WHEN** the checkpointer's database driver binding is not installed
- **THEN** importing the checkpointer SHALL fail with a diagnostic naming the missing driver

#### Scenario: No placeholder alias remains
- **WHEN** the checkpointer module is imported successfully
- **THEN** the checkpointer type SHALL be the real checkpointer type and not a permissive placeholder

#### Scenario: Setup never returns an absent checkpointer
- **WHEN** checkpointer setup is called
- **THEN** it SHALL either return a usable checkpointer or raise, and SHALL NOT return an absent value

### Requirement: The checkpointer consumes a credentialed driver-scheme connection string it does not build itself
The checkpointer SHALL obtain its connection string from the shared accessor for its own driver's flavour, and SHALL
NOT derive, repair, or re-scheme any other flavour's string at its call site. The string it consumes SHALL use the
checkpointer driver's own URL scheme, SHALL carry the configured credentials, and SHALL retain the
transport-security parameters that driver requires. The accessor serving the application's relational engine SHALL
NOT be reused for the checkpointer, because that accessor returns the relational engine's dialect alias, which the
checkpointer driver cannot parse. No accessor and no checkpointer code path SHALL emit the connection string, or any
part of its credentials, to logs.

The accessor set itself — one accessor per flavour, with credential injection and scheme repair living only there —
is defined by `infrastructure-client-access` in the foundation change. This capability is that accessor's consumer,
and it is the reason the plain driver flavour exists: the checkpointer is the consumer that can parse neither the raw
configured URL (which carries no password) nor the relational engine's dialect-aliased form.

#### Scenario: The checkpointer consumes the accessor rather than transforming a string
- **WHEN** the checkpointer is constructed
- **THEN** its connection string SHALL have come from the accessor for its driver's flavour, and no scheme repair or credential injection SHALL occur at the checkpointer's call site

#### Scenario: The consumed string is a credentialed driver-scheme URL
- **WHEN** the checkpointer connection string is requested
- **THEN** it SHALL use the checkpointer driver's scheme, SHALL contain the configured credentials, and SHALL retain the transport-security parameters that driver requires

#### Scenario: The relational engine dialect alias is never passed to the checkpointer
- **WHEN** the checkpointer is constructed
- **THEN** its connection string SHALL NOT contain the relational engine's dialect alias

#### Scenario: The connection string is never logged
- **WHEN** checkpointer setup succeeds or fails
- **THEN** no log record SHALL contain the connection string or its credentials
