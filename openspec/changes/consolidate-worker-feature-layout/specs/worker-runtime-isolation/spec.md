## Purpose

Keep task producers independent of worker resources while preserving durable message compatibility and predictable async execution across repeated deliveries.

## ADDED Requirements

### Requirement: Dispatch validation is available without worker initialization
The application SHALL validate all declared task payloads and reject unknown task names without loading task implementations or starting worker resources. Invalid payloads SHALL fail before publication with the task name and original validation detail.

#### Scenario: Producer starts in a fresh process
- **WHEN** a fresh process imports the producer and validates a document ingestion payload
- **THEN** no worker implementation or feature task module is imported and no network connection is opened

#### Scenario: Invalid dispatch is refused
- **WHEN** a producer submits a missing field, unexpected field, or undeclared task name
- **THEN** it receives a named validation failure and the broker receives no message

### Requirement: Repeated task execution retains async resource ownership
Async resources cached by a worker SHALL execute and close on their owning event loop in the current process. Resource creation SHALL occur after fork, and slow capability provisioning SHALL NOT run inside the child initialization signal.

#### Scenario: Repeated reliability operations
- **WHEN** two successive tasks use the cached async reliability client
- **THEN** both operations use the same live loop

#### Scenario: Child startup and first capability use
- **WHEN** a worker child starts and subsequently executes its first browser or ingestion task
- **THEN** startup does not provision the capability and first use initializes it in that child on its persistent loop

#### Scenario: Failed provisioning and orderly shutdown
- **WHEN** provisioning partially fails or the worker shuts down normally
- **THEN** constructed resources are closed on their owning loop, remaining cleanup is attempted, and repeated shutdown is safe

### Requirement: Source relocation preserves durable delivery compatibility
The migration SHALL preserve all existing task wire names, payload fields, queue destinations, retry policies and public HTTP behavior. Every deployed worker entry point SHALL register all task implementations explicitly.

#### Scenario: Old message reaches the relocated consumer
- **WHEN** a message uses any pre-migration task name and valid payload
- **THEN** the relocated worker recognizes it and routes it to the same queue

#### Scenario: Scheduled memory consolidation
- **WHEN** the scheduler emits memory consolidation
- **THEN** its arguments match the consumer's keyword-only signature

### Requirement: Automated verification rejects runtime boundary regressions
The standard verification workflow SHALL reject imports from producer/HTTP code into worker execution and task entry points, and SHALL detect feature task modules missing from explicit registration.

#### Scenario: Producer acquires a worker dependency
- **WHEN** a worker implementation dependency is introduced into producer code
- **THEN** the boundary check fails

#### Scenario: New feature task is not registered
- **WHEN** a feature task module is added without inclusion in the worker registry
- **THEN** registration verification fails
