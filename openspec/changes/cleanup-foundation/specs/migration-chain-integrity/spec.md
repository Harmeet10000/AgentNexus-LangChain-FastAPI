## Purpose

Guarantee that the schema migration chain has exactly one head, that the target relational schema is defined in
full by exactly one authoritative revision, and that every relation the running application reads or writes
actually exists after the deployed database is upgraded.

## ADDED Requirements

### Requirement: The migration chain SHALL expose exactly one head

The revision graph SHALL present a single head. Where two lineages branched, they SHALL be joined by a revision
that has both as parents, so that upgrading, rendering and schema-comparison commands have an unambiguous target.

#### Scenario: Heads are enumerated

- **WHEN** the migration chain's heads are enumerated
- **THEN** exactly one head SHALL be reported

#### Scenario: Rendering an upgrade to head

- **WHEN** an upgrade to head is rendered without applying it
- **THEN** the render SHALL succeed
- **AND** it SHALL NOT fail because more than one head is present

#### Scenario: The historical branch point is preserved

- **WHEN** the revision graph is inspected after the join
- **THEN** the original branch point SHALL still be reachable in the history
- **AND** no pre-existing revision's identity or parentage SHALL have been rewritten

### Requirement: Exactly one revision SHALL be authoritative for the target schema

A single revision SHALL define the target relational schema in full: the unified document and chunk relations
with their uniqueness constraints, their vector, keyword and fuzzy retrieval indexes, the extensions those
indexes require, and the event-outbox relations. Anyone reading that one revision SHALL be able to see the whole
target shape without reconstructing it from earlier revisions.

That revision SHALL be idempotent and non-destructive: applying it to a database where an earlier revision
already created part of the target schema SHALL converge on the same final shape, SHALL NOT fail, and SHALL NOT
drop or truncate an existing relation.

#### Scenario: Applied to a database that has none of the target relations

- **WHEN** the authoritative revision is applied to a database that holds none of the target relations
- **THEN** the document, chunk and event-outbox relations SHALL exist afterwards with the target shape,
  constraints and indexes

#### Scenario: Applied to a database where an earlier revision already created the document relations

- **WHEN** the authoritative revision is applied to a database in which an earlier revision already created the
  document and chunk relations
- **THEN** the revision SHALL complete without error
- **AND** every row already stored in those relations SHALL still be present afterwards
- **AND** the final shape SHALL match the target, including any column the earlier revision omitted

#### Scenario: Extensions the target indexes depend on

- **WHEN** the authoritative revision is applied
- **THEN** the extensions required by the vector, keyword-ranking and fuzzy-matching indexes SHALL be created if
  they are not already installed
- **AND** applying the revision to a database where they are already installed SHALL NOT fail

#### Scenario: Ordering within the revision

- **WHEN** the authoritative revision is applied
- **THEN** the event-outbox relations SHALL be created before the document schema, so that a failure in the
  larger half cannot prevent the repair of the public endpoints that depend on the outbox

### Requirement: Every relation the running application reads or writes SHALL exist after an upgrade

Upgrading the deployed database to head SHALL leave no live read or write path pointing at a relation that does
not exist. In particular the event-outbox relations SHALL exist, because mounted public endpoints write to them
inside the same transaction that mutates user state.

#### Scenario: A public endpoint that records a domain event

- **WHEN** a mounted public endpoint that records a domain event is called after the deployed database has been
  upgraded to head
- **THEN** the event row SHALL be recorded and the endpoint SHALL return its normal response
- **AND** it SHALL NOT fail with an undefined-relation error after having already persisted user state

#### Scenario: The event relay's startup scan

- **WHEN** the application starts after the upgrade and its event relay scans for unpublished events
- **THEN** the relation it scans SHALL exist
- **AND** the scan SHALL complete rather than being abandoned by its degradation handler

#### Scenario: No live path is left pointing at a missing relation

- **WHEN** the relations named by live read and write paths are compared against the upgraded database
- **THEN** every such relation SHALL be present

### Requirement: Stored chunks SHALL record when they were last written

The chunk relation SHALL carry a last-written timestamp that is never null and is supplied by the database when
a writer does not set it. Without it, a re-embedding campaign cannot distinguish a current-generation embedding
from one carried over from an earlier generation.

#### Scenario: The chunk relation exposes a last-written timestamp

- **WHEN** the deployed database has been upgraded to head
- **THEN** the chunk relation SHALL have a non-null `updated_at` timestamp column carrying a time zone
- **AND** inserting a chunk without supplying that column SHALL succeed, with the database supplying the value

### Requirement: The authoritative revision SHALL NOT claim relations an earlier revision already claims

Reversing the authoritative revision SHALL NOT remove relations whose creation an earlier revision also claims,
because that earlier revision's own reversal already removes them. A reversal that dropped them twice would fail
on a relation that no longer exists and would corrupt the earlier revision's contract.

#### Scenario: Reversing the authoritative revision

- **WHEN** the authoritative revision is reversed
- **THEN** the event-outbox relations SHALL be left in place
- **AND** the reversal SHALL state why, so the next reader does not read the omission as an oversight

### Requirement: A fresh environment SHALL have a documented, repeatable route to the target schema

Some revisions are recorded as applied in the deployed database while having created nothing, and they SHALL NOT
be rewritten. Because of that, a plain upgrade from an empty database is not a supported route to the target
schema. The change record SHALL therefore identify those revisions by name and SHALL document one repeatable
procedure that brings a fresh environment to the target schema without editing any existing revision, naming the
revisions the procedure deliberately skips and what each of them would have created.

#### Scenario: Preparing a new environment

- **WHEN** an operator prepares a new environment from an empty database
- **THEN** a documented procedure SHALL bring that database to the target schema
- **AND** the procedure SHALL name the revisions it deliberately skips

#### Scenario: Reversing below the joined head

- **WHEN** reversing the chain below the joined head is considered
- **THEN** the change record SHALL identify it as unsupported, because the reversals of the revisions recorded
  as applied would drop relations that were never created
