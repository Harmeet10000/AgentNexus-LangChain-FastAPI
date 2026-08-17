## Purpose

Define the persistence and resumability contract for the ingestion pipeline: a checkpointer that is actually
constructible against a real database driver, an application-owned connection pool with a teardown that closes
it, thread identity tied to the document, and checkpointed state that carries references rather than document
payloads.

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

### Requirement: The application owns the checkpointer connection pool and closes it on shutdown
The application SHALL create and own the connection pool the checkpointer uses, and shutdown SHALL close that
pool. Teardown SHALL NOT be a silent no-op, and SHALL succeed without error when no checkpointer was created.

#### Scenario: Shutdown closes the pool
- **WHEN** the application shuts down with a checkpointer active
- **THEN** the connection pool SHALL be closed and the closure SHALL be observable

#### Scenario: Teardown with no checkpointer is a clean no-op
- **WHEN** the application shuts down without having created a checkpointer
- **THEN** teardown SHALL complete without raising

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

### Requirement: The checkpointer connection string carries credentials in its driver's own scheme
The system SHALL provide a connection-string accessor for the checkpointer that uses the checkpointer driver's own
URL scheme, carries the configured credentials, and retains the transport-security parameters that driver
requires. The accessor used by the application's relational engine SHALL NOT be reused for the checkpointer, and
no accessor SHALL emit the connection string to logs.

#### Scenario: The accessor returns a credentialed driver-scheme URL
- **WHEN** the checkpointer connection string is requested
- **THEN** it SHALL use the checkpointer driver's scheme and SHALL contain the configured credentials

#### Scenario: The relational engine dialect alias is never passed to the checkpointer
- **WHEN** the checkpointer is constructed
- **THEN** its connection string SHALL NOT contain the relational engine's dialect alias

#### Scenario: The connection string is never logged
- **WHEN** checkpointer setup succeeds or fails
- **THEN** no log record SHALL contain the connection string or its credentials

### Requirement: Consumers of a deliberately unprovisioned checkpointer fail closed
Where the deployment deliberately does not provision the shared checkpointer, every consumer that reads it SHALL
fail closed with a typed service-unavailable error naming the missing capability, and SHALL NOT raise an
attribute error on an absent value.

#### Scenario: An agent request without a checkpointer returns service unavailable
- **WHEN** a request reaches a consumer requiring the shared checkpointer while it is absent
- **THEN** the system SHALL respond with a typed service-unavailable error

#### Scenario: The absent checkpointer never surfaces as an internal error
- **WHEN** the shared checkpointer is absent
- **THEN** no consumer SHALL raise an attribute error and no request SHALL fail with an unhandled internal error

### Requirement: Each database URL flavour has exactly one accessor
The system SHALL expose exactly one accessor per database URL flavour it needs, and every consumer of a flavour
SHALL obtain its URL from that accessor. The credential injection and scheme repair for a flavour SHALL NOT be
reimplemented at a call site.

#### Scenario: Every consumer of a flavour uses its accessor
- **WHEN** any component needs a database connection string
- **THEN** it SHALL obtain it from the accessor for the flavour it requires rather than transforming another flavour's string itself

#### Scenario: No consumer can obtain an unusable URL
- **WHEN** a connection string is obtained from any accessor
- **THEN** it SHALL carry the configured credentials and a scheme its intended driver accepts

#### Scenario: Scheme repair is not duplicated
- **WHEN** the repository is inspected for transformations of the configured database URL
- **THEN** each transformation SHALL occur in exactly one accessor rather than being repeated at consumer sites
