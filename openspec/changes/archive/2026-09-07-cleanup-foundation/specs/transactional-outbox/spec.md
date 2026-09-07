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

<!--
Deliberately NOT added here: a requirement that a missing outbox relation must fail loudly instead of being
absorbed by the relay's broad exception handling.

That requirement was drafted for this change and has been withdrawn, because every clause of it is a change to the
relay's exception handling and to the auth service's transaction boundary — work this change disclaims in three
places, ships no step for, and deliberately sequences after the relations exist. It would also have contradicted an
accepted requirement this change does not modify: `typed-exception-handling`'s *Degradation boundaries SHALL keep
except Exception with add_note*, whose scenario *Outbox relay dead-letters on any failure* currently sanctions the
broad catch. Two accepted specs disagreeing about the same lines is worse than one recorded gap.

The gap is therefore carried as an explicit Non-Goal in `design.md`, and the decision about what the relay owes when
a relation is absent — including which spec wins until the narrowing lands, and the paired
`typed-exception-handling` MODIFIED that the narrowing change must ship — is recorded in `adrs.md` as ADR-5.
-->

