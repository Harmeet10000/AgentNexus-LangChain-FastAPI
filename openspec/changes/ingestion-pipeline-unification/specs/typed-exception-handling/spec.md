## ADDED Requirements

### Requirement: Embedding failures SHALL raise a typed failure rather than substitute a placeholder value
Embedding operations SHALL raise a typed project exception when the provider fails, adds context via
`exc.add_note()` naming the model, task type, and text count, and SHALL NOT return or persist a placeholder vector
of any kind. A placeholder vector is a valid row that ranks against nothing, which makes the failure invisible to
every downstream consumer.

#### Scenario: Provider failure raises with context
- **WHEN** an embedding provider call fails
- **THEN** a typed project exception SHALL be raised, carrying the provider's own exception as its cause and a note naming the model, task type, and text count

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

### Requirement: Retry boundaries SHALL raise one typed transient failure and their callers SHALL be converted to catch it
A retry boundary that exhausts its attempts SHALL raise the project's typed transient-failure exception, chained to
the original exception so the original type, message, and traceback remain reachable as that exception's cause.
Chaining preserves the *cause*, not the raised *type*: a caller catching the provider's or framework's own base
exception type will therefore **not** match the transient-failure exception, however it is chained. Every caller with
a degradation branch around a retried operation SHALL therefore be converted to catch the transient-failure type as
well, and no such branch SHALL be left matching only a type the boundary can no longer raise.

#### Scenario: Exhausted retries chain the original cause
- **WHEN** a retry boundary exhausts its attempts
- **THEN** the raised transient-failure exception SHALL carry the original exception as its cause, and the original exception's type and message SHALL be recoverable from that cause

#### Scenario: Every caller's degradation branch matches what the boundary raises
- **WHEN** the callers that wrap a retried operation in a degradation branch are inspected
- **THEN** each SHALL catch the transient-failure type raised by the retry boundary, and none SHALL rely solely on catching a provider or framework base type that the boundary does not raise

#### Scenario: A degradation branch fires for an exhausted retry
- **WHEN** a retried operation inside a converted caller exhausts its attempts
- **THEN** that caller's degradation branch SHALL execute, and the recorded diagnostic SHALL name the original failure reached through the chained cause

#### Scenario: Distinct upstream failures remain distinguishable
- **WHEN** authentication, quota, and malformed-response failures occur inside a retry boundary
- **THEN** each SHALL remain distinguishable by its chained cause rather than collapsing into one opaque failure with no recoverable origin

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
