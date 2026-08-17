## Purpose

Guarantee that every persisted model live code depends on is visible to the single schema registry that
migration authoring compares against the database, so that generated migrations can never propose dropping a
relation the application still uses.

## ADDED Requirements

### Requirement: Every live persisted model SHALL be registered on the single shared schema registry

The registry that migration authoring compares against the database SHALL contain every relation that live code
reads or writes. A persisted model declared against its own private registry SHALL be re-declared on the shared
one or removed; leaving it on a private registry makes it invisible to comparison while still being live in the
application.

#### Scenario: The shared registry is enumerated

- **WHEN** the shared schema registry is enumerated after this change
- **THEN** it SHALL include the unified document and chunk relations, the search relations, the event-outbox
  relations and the billing relations

#### Scenario: A model declared against a private registry

- **WHEN** a persisted model that live code depends on is found declared against a registry other than the
  shared one
- **THEN** that model SHALL be re-declared on the shared registry
- **AND** merely importing its module SHALL NOT be accepted as registration, because importing a module that
  owns a private registry registers nothing on the shared one

### Requirement: Schema comparison SHALL never propose removing a live relation

Comparing the shared registry against the database SHALL NOT propose dropping a relation that live code reads or
writes. Registration exists to make that comparison trustworthy; it is not an endorsement of the relations it
makes visible, and it SHALL NOT be used as a licence to generate the target schema by comparison.

#### Scenario: Comparison after registration is complete

- **WHEN** the registry is compared against a database that holds the live relations
- **THEN** no live relation SHALL appear as a candidate for removal

#### Scenario: Registration of relations the database does not yet hold

- **WHEN** a relation is registered while the database does not yet hold it
- **THEN** the creation of that relation SHALL remain the responsibility of the authoritative target-schema
  revision
- **AND** it SHALL NOT be produced by generating a migration from the comparison

### Requirement: Incomplete registration SHALL fail loudly

If a model module the migration environment names cannot be imported, the migration command SHALL fail with that
import error. It SHALL NOT continue with a partially populated registry, because a silently incomplete registry
is exactly the condition under which comparison proposes destructive changes.

#### Scenario: A named model module cannot be imported

- **WHEN** the migration environment names a model module that cannot be imported
- **THEN** the migration command SHALL fail and surface the import error

#### Scenario: No unreachable fallback conceals the failure

- **WHEN** the migration environment's error handling is inspected
- **THEN** it SHALL NOT contain a fallback path that cannot be reached and therefore reports nothing
