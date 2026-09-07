## MODIFIED Requirements

### Requirement: Database operations SHALL catch asyncpg.exceptions.PostgresError

All asyncpg operations SHALL catch `asyncpg.exceptions.PostgresError` or its subclasses instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with the query, table, and operation context.

Client-side errors (`asyncpg.InterfaceError`, `asyncpg.InternalClientError`) SHALL be caught separately when they indicate programming errors rather than database failures.

**All six scenarios of the accepted requirement are reproduced below, and the first one is reproduced deliberately
even though this change deletes the code it describes.** An earlier draft of this delta reproduced only five, omitting
`Reconciliation fetch failure catches PostgresError` on the reasoning that a scenario constraining deleted code is
unfalsifiable. That draft was wrong about the *mechanism*, and the mechanism decides the outcome:

- **A `## MODIFIED Requirements` block replaces its target requirement wholesale on archive.** It is not diffed and
  not merged. So a block reproducing five of six scenarios does not leave the sixth alone — it **deletes** it from the
  deployed spec, with no `## REMOVED` block, no Reason and no Migration in the archived record.
- **`validate --strict` cannot detect it.** The block is structurally well-formed on its own; the evidence that a
  scenario is missing lives in a file this change does not contain. A green validation is therefore not evidence that
  a MODIFIED block is complete.
- **`## REMOVED Requirements` is not the alternative.** It operates at requirement granularity: naming this
  requirement there would retire the entire asyncpg guarantee, including the five scenarios that must survive. There
  is no scenario-level REMOVED.

So the scenario is **kept, verbatim, in its original position**, and its retirement is routed rather than performed
here. Retiring it is a direct edit to the accepted spec — the same kind of edit the four `## Purpose` failures in
`openspec validate --all` need, and it belongs with them, in a spec-hygiene pass rather than inside a schema and
subtraction change. Until then the scenario describes a path that no longer exists, which is a visible, correctable
staleness rather than a guarantee that vanished with no author.

#### Scenario: Reconciliation fetch failure catches PostgresError
- **WHEN** a reconciliation database query fails
- **THEN** the code catches `asyncpg.exceptions.PostgresError`, adds a note with the user_id and query, and returns a failure result

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
