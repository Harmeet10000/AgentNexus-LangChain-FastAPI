## MODIFIED Requirements

### Requirement: Database operations SHALL catch asyncpg.exceptions.PostgresError

All asyncpg operations SHALL catch `asyncpg.exceptions.PostgresError` or its subclasses instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with the query, table, and operation context.

Client-side errors (`asyncpg.InterfaceError`, `asyncpg.InternalClientError`) SHALL be caught separately when they indicate programming errors rather than database failures.

#### Scenario: Outbox publish failure catches PostgresError
- **WHEN** an outbox event publish fails at the database level
- **THEN** the code catches `asyncpg.exceptions.PostgresError`, adds a note with the event_id and event_type, and marks the event as failed

#### Scenario: Unique violation catches UniqueViolationError
- **WHEN** an INSERT/UPDATE violates a UNIQUE constraint
- **THEN** the code catches `asyncpg.exceptions.UniqueViolationError`, adds a note with the constraint name, and raises ConflictException

#### Scenario: Connection failure catches ConnectionDoesNotExistError
- **WHEN** a query fails because the connection was closed/pooled away
- **THEN** the code catches `asyncpg.exceptions.ConnectionDoesNotExistError`, adds a note with the operation, and retries with a new connection

#### Scenario: Deadlock detected catches DeadlockDetectedError
- **WHEN** a query fails because of a deadlock
- **THEN** the code catches `asyncpg.exceptions.DeadlockDetectedError`, adds a note with the query, and retries the transaction

#### Scenario: Client misuse catches InterfaceError
- **WHEN** an asyncpg API is used incorrectly (closed connection, wrong call order)
- **THEN** the code catches `asyncpg.exceptions.InterfaceError`, adds a note with the operation, and raises DatabaseException (programming error, not retryable)
