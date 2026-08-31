## Purpose

Guarantees that a repository which catches a database exception leaves the session
usable, so that a failure a caller chooses not to propagate can never be followed
by a successful commit of a partial write, and states what a repository owes when its
backing store has no transaction to roll back.

## ADDED Requirements

### Requirement: A relational repository SHALL roll back before returning a failure from a caught database exception

When a repository catches an exception raised by the SQLAlchemy driver or ORM after
any statement has been sent on the request-scoped session, it SHALL roll back that
session before returning `Failure`. The order SHALL be: classify the exception into
the feature's error type, roll back, log, return.

Rollback SHALL happen in the repository, not in the service and not in the request
dependency, because the repository is the only layer that knows a statement was
issued.

#### Scenario: A flush failure leaves the session usable
- **WHEN** a repository catches `IntegrityError` from a flush and returns its conflict error
- **THEN** it has rolled back the session first, and a subsequent statement on that same session succeeds rather than raising `PendingRollbackError`

#### Scenario: The rollback precedes the log line
- **WHEN** a repository catches `SQLAlchemyError` and both rolls back and logs
- **THEN** the rollback happens before the log call, so a logging failure cannot leave the session poisoned

#### Scenario: A read-only failure needs no rollback
- **WHEN** a repository catches an exception from a `SELECT` that issued no write
- **THEN** it may return `Failure` without rolling back, and the requirement is satisfied by the session remaining usable

### Requirement: A swallowed failure SHALL NOT be followed by a commit of partial work

A caller that receives a `Failure` and neither propagates it nor aborts the unit
of work SHALL NOT allow that request to reach a successful commit carrying the
partial write that failed.

This is the live defect the rollback closes. The request-scoped session commits on
successful exit and rolls back only when an exception escapes —
`get_postgres_db` at `src/app/connections/postgres.py:241` yields, then calls
`await session.commit()`, with `except Exception: await session.rollback(); raise` as
its only rollback path. A returned `Failure` is not an escaping exception. Where a
service unwraps a `Failure` and continues — the webhooks service unwraps repository
Results at 21 sites and calls no bridge; the dunning service does the same — no
exception escapes, so the dependency's rollback path never runs, and `commit()`
executes on a session whose transaction is already failed.

The dependency cannot detect that condition: it sees no exception and has no access
to the `Result`. Widening it is therefore not an alternative to this requirement;
`shared-infrastructure-errors` states that constraint from the dependency's side.

#### Scenario: A swallowed repository failure does not commit
- **WHEN** a service receives a `Failure` from a write, logs it, and returns a success envelope to its caller
- **THEN** the partial write from the failed statement is not committed, because the repository already rolled it back

#### Scenario: A multi-step operation fails partway
- **WHEN** the second of three writes in one request fails and the service abandons the operation
- **THEN** the first write is not persisted, and the response does not report success for work that was rolled back

#### Scenario: The failed statement's error does not resurface at commit
- **WHEN** a request continues after a caught database exception and reaches the end of its dependency scope
- **THEN** the commit does not raise a rollback-required error attributed to the earlier statement

### Requirement: Every relational repository SHALL be covered by this rule uniformly

All repositories that issue statements on the request-scoped SQLAlchemy session
SHALL follow the same rollback contract. A repository that catches a database
exception without rolling back SHALL be a rule violation detectable by the project's
enforcement gates, not a matter of reviewer attention.

Measured scope. There are **11 repository modules** holding 12 repository classes
(`auth/repository.py` holds two). Of those:

- **9 are relational** and carry 74 SQLAlchemy handlers between them — `audit`,
  `credits/consumption_repository`, `credits/credit_repository`, `documents`,
  `invoices`, `payments`, `plans`, `subscriptions`, `webhooks`. These are the modules
  this requirement covers. None of them rolls back today; `session.rollback` has
  never appeared under `src/app/features/` in the repository's history.
- **`auth/repository.py` is a document-store repository**, not a relational one. It
  catches `PyMongoError`, `DuplicateKeyError` and `RedisError` across 19 handlers and
  issues no statement on the SQLAlchemy session. It is covered by the next
  requirement instead.
- **`users/repository.py` catches nothing**, so it has no handler to fix.

The rule therefore applies to every existing `except` block in the 9 relational
modules, not only to new code.

Database exceptions are also caught in three places that are not repositories —
`features/health/service.py` and the two agent tools
`shared/langchain_layer/agents/tools/{retrieve_statute_section,search_legal_precedents}.py`.
All three are read-only and issue no write, so the read-only scenario above already
covers them and their absence from a repository-scoped gate is not a hole.

#### Scenario: A new repository inherits the obligation
- **WHEN** a relational repository is added with an `except SQLAlchemyError` block that returns `Failure` without rolling back
- **THEN** the enforcement gate reports a violation

#### Scenario: A migrated feature has no unrolled-back handler
- **WHEN** the enforcement gate is run over a feature whose migration change is complete
- **THEN** it reports zero database-exception handlers that return `Failure` without rolling back

#### Scenario: A non-repository read-only catcher is not a violation
- **WHEN** the gate encounters a health probe or an agent tool that catches `SQLAlchemyError` on a read and returns a status dict or an unavailable tool result
- **THEN** no violation is reported, because no statement was written and the session remains usable

### Requirement: A document-store repository SHALL classify its driver taxonomy and SHALL NOT be given a rollback

A repository whose backing store has no request-scoped transaction SHALL classify
its driver's exceptions into its feature's union and return `Failure`, and SHALL NOT
be required to roll back. It SHALL NOT be exempted from classification on the grounds
that the rollback rule does not reach it.

`auth/repository.py` is the only instance. It holds `UserRepository` and
`RefreshTokenRepository`, backed by MongoDB and Redis, with 13 Mongo handlers and 6
Redis handlers and no SQLAlchemy usage. There is no session to poison, so the
poisoned-commit defect cannot occur there — but its **7 `"DB_ERROR"` literals** are
Mongo and Redis failures wearing a relational label, and they retire on the same
terms as the other 49.

Their retryability differs and SHALL be preserved. A failed relational transaction is
dead and the correction sends it from 503 to 500. A Mongo or Redis outage is
genuinely retryable, so those 7 sites SHALL keep a retryable classification and SHALL
NOT be swept into the 503 → 500 correction.

#### Scenario: A document-store failure is classified without a rollback
- **WHEN** `UserRepository` catches `PyMongoError` from a write
- **THEN** it classifies into the auth feature's union and returns `Failure`, and no rollback is attempted or required

#### Scenario: A duplicate key becomes a conflict, not an infrastructure failure
- **WHEN** `UserRepository` catches `DuplicateKeyError` on an insert
- **THEN** it returns the union's conflict member with `ErrorKind.CONFLICT`, not an infrastructure error

#### Scenario: The retryable classification survives the DB_ERROR correction
- **WHEN** the 7 `"DB_ERROR"` literals in `auth/repository.py` are replaced by enum members
- **THEN** their kind remains retryable infrastructure and their status remains 503, distinct from the relational sites that correct to 500
