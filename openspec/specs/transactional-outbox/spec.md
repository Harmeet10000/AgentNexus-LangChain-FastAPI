# transactional-outbox Specification

## Purpose
TBD - created by archiving change celery-outbox-idempotency. Update Purpose after archive.
## Requirements
### Requirement: Outbox Table Schema
#### Scenario: Table exists
- **WHEN** migration runs
- **THEN** `outbox_events` and `dead_letter_events` tables exist

### Requirement: Outbox Helper
#### Scenario: Helper inserts row
- **WHEN** `with_outbox()` is called
- **THEN** an outbox row is inserted and `pg_notify` fires

### Requirement: Relay Process
#### Scenario: Notification processed
- **WHEN** a notification arrives
- **THEN** the relay publishes the event or dead-letters after 5 failures

### Requirement: Relay Lifecycle
#### Scenario: Lifecycle managed
- **WHEN** the app starts
- **THEN** the relay starts after deps are ready and drains on shutdown

### Requirement: Dead Letter
#### Scenario: Dead letter
- **WHEN** publish fails 5 times
- **THEN** the event is moved to `dead_letter_events`

### Requirement: Migration
#### Scenario: Migration runs
- **WHEN** alembic upgrade is run
- **THEN** both tables are created idempotently

