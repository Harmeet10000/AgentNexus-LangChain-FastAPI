## Purpose

Define the deployment contract that makes asynchronous ingestion real: a process consumes the queue, a scheduler
runs periodic work, the documented start command loads the application, task registration does not depend on an
import side effect, and task names are defined once.

## ADDED Requirements

### Requirement: The deployment runs a process that consumes the task queue
The deployment SHALL include a worker process that consumes the task queues the application dispatches to. A
deployment in which no process consumes the queue SHALL NOT be considered complete.

#### Scenario: A dispatched ingestion task is executed
- **WHEN** an ingestion task is dispatched to the queue
- **THEN** a worker process SHALL consume and execute it

#### Scenario: The worker reports the ingestion task as registered
- **WHEN** the running worker is asked which tasks it has registered
- **THEN** the ingestion task name SHALL appear in the reply

#### Scenario: The worker consumes the queues ingestion is routed to
- **WHEN** the running worker is asked which queues it consumes
- **THEN** the queues ingestion tasks are routed to SHALL appear in the reply

### Requirement: The deployment runs a scheduler for periodic tasks
The deployment SHALL include a scheduler process that emits the configured periodic tasks, and those tasks SHALL
be executed by a worker.

#### Scenario: A scheduled task fires and is executed
- **WHEN** a configured periodic task's schedule elapses
- **THEN** the scheduler SHALL emit it and a worker SHALL execute it

#### Scenario: The scheduler is present in the deployment
- **WHEN** the deployment's services are enumerated
- **THEN** a scheduler process SHALL be present alongside the worker

### Requirement: The documented worker start command loads the application
The documented command for starting a worker SHALL reference the application's real task application and SHALL
start successfully. It SHALL match the command the deployed worker process uses.

#### Scenario: The documented command starts a worker
- **WHEN** the documented worker start command is run
- **THEN** the worker SHALL start and report its registered tasks, and SHALL NOT fail to load the application

#### Scenario: No stale application reference remains
- **WHEN** the repository is searched for the worker start command's application reference
- **THEN** every occurrence SHALL name a module that exists

### Requirement: Task registration is explicit, not an import side effect
Every task module the system dispatches to SHALL be registered explicitly in the task application's module list.
Registration SHALL NOT depend on one module being imported as a side effect of importing another, including
modules that are declared but not yet implemented.

#### Scenario: Registration survives an unrelated package edit
- **WHEN** unrelated imports are removed from the task package's initialiser
- **THEN** the dispatched task names SHALL still be registered

#### Scenario: Every dispatched task module is listed
- **WHEN** the task application's module list is inspected
- **THEN** it SHALL include every module containing a task the system dispatches to

#### Scenario: A declared but unimplemented task is registered
- **WHEN** a task module exists whose task is deliberately not implemented
- **THEN** the task SHALL still be registered, and invoking it SHALL fail with an explicit not-implemented error rather than an unknown-task error

### Requirement: Task names are defined once and shared by producer and consumer
Each dispatchable task name SHALL be defined in exactly one place, and both the dispatching side and the task
declaration SHALL reference that definition. Dispatch to a name that is not registered SHALL fail with a
diagnostic naming the task rather than being discarded.

#### Scenario: Producer and consumer share one name definition
- **WHEN** a task name is changed at its single definition
- **THEN** both the dispatching side and the task declaration SHALL follow, with no string literal left behind

#### Scenario: Dispatch to an unregistered name is reported
- **WHEN** an event dispatches a task name that is not registered
- **THEN** the system SHALL record a failure naming the task rather than silently discarding the dispatch

### Requirement: Long-running ingestion work does not starve latency-sensitive tasks
Ingestion tasks, whose duration is measured in minutes, SHALL be routed such that they cannot delay the execution
of latency-sensitive tasks such as billing and transactional email beyond those tasks' expected latency.

#### Scenario: A long ingestion does not delay a short task
- **WHEN** several minutes-long ingestion tasks are executing
- **THEN** a newly dispatched latency-sensitive task SHALL begin executing without waiting for them to finish

#### Scenario: Routing is explicit for every dispatched task
- **WHEN** the routing configuration is inspected
- **THEN** every dispatched task name SHALL resolve to an explicitly configured destination rather than an implicit default

### Requirement: Worker readiness is verifiable independently of the dispatch chain
The presence of a queue consumer SHALL be verifiable by interrogating the running worker directly, without
depending on an upstream dispatch chain succeeding. Verification SHALL NOT require that a durable outbound event
be recorded, relayed, and delivered first.

#### Scenario: Consumer presence is proven by interrogating the worker
- **WHEN** an operator needs to confirm that the queue has a consumer
- **THEN** interrogating the running worker for its registered tasks and consumed queues SHALL be sufficient, with no upstream event required

#### Scenario: A broken dispatch chain does not hide a present worker
- **WHEN** the durable outbound event store or its relay is unavailable
- **THEN** the worker's readiness SHALL still be observable, and its status SHALL NOT be reported as unknown

### Requirement: A silent link in the dispatch chain is not acceptable
Each stage of the path from accepting a document to executing its ingestion task SHALL report its own failure
distinctly. A failure at one stage SHALL NOT be reported only as an absence of downstream effect.

#### Scenario: Each stage reports its own failure
- **WHEN** any single stage of the dispatch path fails
- **THEN** that stage SHALL emit a failure record identifying itself, distinguishable from a failure of any other stage

#### Scenario: An absent downstream effect is diagnosable
- **WHEN** an accepted document is never processed
- **THEN** the recorded diagnostics SHALL identify which stage of the dispatch path did not complete
