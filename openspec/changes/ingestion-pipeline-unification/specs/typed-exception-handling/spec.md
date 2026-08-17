## ADDED Requirements

### Requirement: Embedding failures SHALL raise a typed failure rather than substitute a placeholder value
Embedding operations SHALL raise a typed project exception when the provider fails, adds context via
`exc.add_note()` naming the model, task type, and text count, and SHALL NOT return or persist a placeholder vector
of any kind. A placeholder vector is a valid row that ranks against nothing, which makes the failure invisible to
every downstream consumer.

#### Scenario: Provider failure raises with context
- **WHEN** an embedding provider call fails
- **THEN** the code catches the provider's own exception type, adds a note naming the model, task type, and text count, and raises a typed project exception

#### Scenario: No placeholder vector is returned
- **WHEN** an embedding operation cannot produce a vector
- **THEN** it SHALL NOT return a zero-filled or otherwise synthesised vector to its caller

#### Scenario: No placeholder vector is persisted
- **WHEN** an embedding failure occurs during ingestion
- **THEN** no chunk record SHALL be persisted for the affected text, and the document SHALL record the failure

### Requirement: Retry boundaries SHALL retry only named transient exception types
Retry wrappers at input/output client boundaries SHALL name the specific transient exception types they retry.
They SHALL NOT retry on a catch-all exception base, and they SHALL apply a growing wait between attempts rather
than retrying immediately.

#### Scenario: A non-transient exception propagates on the first attempt
- **WHEN** an operation inside a retry boundary raises an exception type that is not in the retryable set
- **THEN** it propagates on the first attempt with no further attempts made

#### Scenario: A transient exception is retried with a growing wait
- **WHEN** an operation inside a retry boundary raises a named transient exception type
- **THEN** it is retried up to the configured attempt count, with the wait between attempts increasing

#### Scenario: A catch-all retry predicate is not used
- **WHEN** a retry boundary's retryable set is inspected
- **THEN** it SHALL name specific exception types and SHALL NOT be the base exception type

### Requirement: Retry boundaries SHALL preserve the original exception for callers
A retry boundary SHALL leave the original exception type observable to callers. Where retries are exhausted and a
transient-failure exception is raised instead, it SHALL be chained to the original via `raise ... from exc` so the
original type and message remain reachable, and SHALL NOT be substituted in a way that prevents a caller's
existing degradation branch from matching.

#### Scenario: Exhausted retries chain the original cause
- **WHEN** a retry boundary exhausts its attempts
- **THEN** the raised transient-failure exception carries the original exception as its cause

#### Scenario: A caller's degradation branch still matches
- **WHEN** a caller catches the provider's or framework's own base exception type around a retried operation
- **THEN** that catch SHALL still match the failure the retried operation produced

#### Scenario: Distinct upstream failures remain distinguishable
- **WHEN** authentication, quota, and malformed-response failures occur inside a retry boundary
- **THEN** each SHALL remain distinguishable by type or by its chained cause rather than collapsing into one opaque type

### Requirement: Retry boundaries SHALL NOT wrap whole graph nodes or catch control-flow exceptions
Retry wrappers SHALL be applied at input/output client boundaries only, and SHALL NOT wrap a whole graph node.
They SHALL NOT catch or retry exceptions that a framework uses for control flow, such as a pause raised to hand
control back to the caller.

#### Scenario: A control-flow pause propagates without retry
- **WHEN** a control-flow exception used to pause execution is raised inside a retried operation
- **THEN** it propagates immediately to the framework with no retry attempted

#### Scenario: Retry scope is the client call, not the node
- **WHEN** a graph node performs several input/output calls
- **THEN** each call SHALL be retried individually and the node body as a whole SHALL NOT be wrapped in a retry

#### Scenario: Retry accounting is not silently multiplied by replay
- **WHEN** a graph node that performs retried calls is replayed from a checkpoint
- **THEN** repeated side effects SHALL be prevented by an idempotency key rather than by an attempt counter held in node-local state
