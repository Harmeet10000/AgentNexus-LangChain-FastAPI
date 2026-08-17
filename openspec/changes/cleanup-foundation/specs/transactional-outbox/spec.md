# transactional-outbox Specification Delta

## MODIFIED Requirements

### Requirement: Outbox Table Schema

The event-outbox relations SHALL exist in every database the application connects to. The `outbox_events` and
`dead_letter_events` relations MUST be present before any request path that enqueues an event is served, and a
migration chain that has been stamped past the revision declaring them MUST NOT be treated as evidence that they
exist.

#### Scenario: Table exists

- **WHEN** migration runs
- **THEN** `outbox_events` and `dead_letter_events` tables SHALL exist

#### Scenario: Stamped-but-unapplied history does not satisfy this requirement

- **WHEN** the recorded migration version is at or beyond the revision that declares the outbox relations, but the
  relations are absent from the database
- **THEN** the system SHALL be considered in violation of this requirement
- **AND** a later revision SHALL create the missing relations rather than assuming the recorded version is truthful

#### Scenario: Public endpoints that enqueue events succeed

- **WHEN** a request is made to an endpoint whose success depends on enqueuing an outbox event, such as requesting
  a password reset or re-requesting an email verification
- **THEN** the request SHALL complete successfully rather than failing after its own state change has been
  persisted
- **AND** the enqueued event SHALL be durably recorded in the same transaction as that state change

### Requirement: Migration

Applying the migration chain SHALL leave both outbox relations present, and SHALL do so idempotently — the same
upgrade MUST succeed whether the relations are absent or already present, since the chain has been stamped past
their declaring revision without that revision's body ever having run.

#### Scenario: Migration runs

- **WHEN** alembic upgrade is run
- **THEN** both tables SHALL be created idempotently

#### Scenario: Creation is conditional, not unconditional

- **WHEN** the revision that creates the outbox relations is applied to a database that already holds them
- **THEN** the upgrade SHALL succeed without error
- **AND** existing rows in those relations SHALL be preserved
- **AND** the supporting indexes and the notification trigger SHALL also be created conditionally, so that a
  partially-created outbox is completed rather than rejected

#### Scenario: Upgrade from the recorded live version reaches a single head

- **WHEN** an upgrade is run against a database whose recorded version is the live stamped revision
- **THEN** the upgrade SHALL resolve to exactly one head
- **AND** SHALL terminate with both outbox relations present

## ADDED Requirements

### Requirement: A missing outbox relation SHALL fail loudly rather than silently disabling the outbox

The relay MUST NOT convert the absence of its own relations into a warning that leaves the process healthy. When
the relations the relay depends on do not exist, the outbox is permanently dead for the lifetime of that process:
nothing retries, no event is ever published, and every endpoint that enqueues an event fails after committing its
own state change. Today that condition is absorbed by broad exception handling in both the startup scan and the
listener — the listener additionally inside a fire-and-forget background task, where nothing observes its
outcome — so the application boots successfully only because those handlers are wide enough to hide a schema
defect. No requirement sanctions that behaviour, and the same breadth would hide any future schema drift
identically.

#### Scenario: Startup with absent outbox relations

- **WHEN** the application starts and the relations the relay reads or writes do not exist
- **THEN** the condition SHALL be reported at error severity, distinguishable from a transient connection failure
- **AND** the report SHALL identify the missing relation by name
- **AND** the outbox subsystem SHALL be recorded as unavailable in a form the readiness surface can observe,
  rather than left indistinguishable from a healthy relay

#### Scenario: Background listener failure is observable

- **WHEN** the long-running notification listener terminates for any reason after startup
- **THEN** the termination SHALL be reported at error severity and SHALL NOT be discarded because it occurred in a
  detached background task
- **AND** the outbox subsystem SHALL be recorded as no longer running

#### Scenario: Endpoints do not report success when the event cannot be durably enqueued

- **WHEN** a request path attempts to enqueue an outbox event and the relations are absent
- **THEN** the request SHALL NOT report success
- **AND** the state change that the event was meant to accompany SHALL NOT remain committed without its event

#### Scenario: Ordering constraint on tightening the relay's exception handling

- **WHEN** the relay's broad exception handling is narrowed so that a missing relation is no longer swallowed
- **THEN** that narrowing SHALL NOT be applied to any environment in which the outbox relations do not yet exist
- **AND** the revision that creates the relations SHALL be applied first, so that the change converts a silent
  permanent degradation into a loud, observable failure rather than into a startup failure
