## MODIFIED Requirements

### Requirement: Database operations SHALL catch asyncpg.exceptions.PostgresError

All asyncpg operations SHALL catch `asyncpg.exceptions.PostgresError` or its subclasses instead of bare `except Exception`. Each catch site SHALL add `exc.add_note()` with the query, table, and operation context.

Client-side errors (`asyncpg.InterfaceError`, `asyncpg.InternalClientError`) SHALL be caught separately when they indicate programming errors rather than database failures.

Where database access is mediated by SQLAlchemy inside a repository, the catch SHALL name the SQLAlchemy exception the driver error is wrapped in — `IntegrityError` for constraint violations, `SQLAlchemyError` otherwise — because the asyncpg exception does not reach the repository unwrapped.

A catch site inside a repository SHALL roll back the session and return `Failure` carrying a member of the feature's error union. It SHALL NOT raise. A catch site outside the domain core — a relay, a script, a graph node holding a raw connection — follows its own layer's convention.

#### Scenario: Reconciliation fetch failure catches PostgresError
- **WHEN** a reconciliation database query fails
- **THEN** the code catches `asyncpg.exceptions.PostgresError`, adds a note with the user_id and query, and returns a failure result

#### Scenario: Outbox publish failure catches PostgresError
- **WHEN** an outbox event publish fails at the database level
- **THEN** the code catches `asyncpg.exceptions.PostgresError`, adds a note with the event_id and event_type, and marks the event as failed

#### Scenario: Unique violation is returned as a conflict, not raised
- **WHEN** an INSERT/UPDATE in a repository violates a UNIQUE constraint
- **THEN** the code catches SQLAlchemy's `IntegrityError`, adds a note with the constraint name, rolls back the session, and returns `Failure` carrying the feature's conflict error whose kind is `CONFLICT`

#### Scenario: Connection failure catches ConnectionDoesNotExistError
- **WHEN** a query fails because the connection was closed/pooled away
- **THEN** the code catches `asyncpg.exceptions.ConnectionDoesNotExistError`, adds a note with the operation, and retries with a new connection

#### Scenario: Deadlock detected catches DeadlockDetectedError
- **WHEN** a query fails because of a deadlock
- **THEN** the code catches `asyncpg.exceptions.DeadlockDetectedError`, adds a note with the query, and retries the transaction

#### Scenario: Client misuse is returned as a non-retryable infrastructure failure
- **WHEN** an asyncpg API is used incorrectly (closed connection, wrong call order) inside a repository
- **THEN** the code catches `asyncpg.exceptions.InterfaceError`, adds a note with the operation, rolls back, and returns `Failure` carrying an infrastructure error that declares itself **not** retryable, because a programming error does not become correct on a second attempt

#### Scenario: A rolled-back write is not advertised as retryable
- **WHEN** a repository returns a failure for a write whose transaction has been rolled back
- **THEN** the error's retryable constant is false, so the boundary renders 500 rather than telling the client to retry a transaction that cannot succeed

### Requirement: Agent tools SHALL catch OS-level and library-specific exceptions

Agent tool operations SHALL catch specific exceptions instead of bare `except Exception`:
- `OSError` (and subclasses `FileNotFoundError`, `PermissionError`) for filesystem operations
- `redis.exceptions.RedisError` for Redis operations
- `langchain_core.exceptions.LangChainException` for LLM operations
- `subprocess.SubprocessError` for subprocess execution failures

Each catch site SHALL add `exc.add_note()` with the command, path, or operation context.

The tool boundary returns the normalised `ToolResult` envelope, whose `error` is a
string because the consumer is a language model rather than a type checker. That
string SHALL be derived from a typed error's code and message, not composed
free-form at the catch site, so the same underlying failure reads identically from
every tool. A dependency that is absent or unreachable SHALL use the envelope's
unavailable form rather than its failure form, because an agent can act on
unavailability and cannot act on "the tool is broken".

#### Scenario: Shell command failure catches OSError
- **WHEN** a shell command execution fails due to an OS error
- **THEN** the code catches `OSError`, adds a note with the command and working directory, and returns a `ToolResult.fail(...)` result

#### Scenario: File not found catches FileNotFoundError
- **WHEN** a file read operation fails because the file doesn't exist
- **THEN** the code catches `FileNotFoundError`, adds a note with the file path, and returns a `ToolResult.fail(...)` result

#### Scenario: Permission denied catches PermissionError
- **WHEN** a file or directory operation fails because of insufficient permissions
- **THEN** the code catches `PermissionError`, adds a note with the path and required permission, and returns a `ToolResult.fail(...)` result

#### Scenario: Redis cache failure catches RedisError
- **WHEN** a tool's Redis cache operation fails
- **THEN** the code catches `redis.exceptions.RedisError`, adds a note with the key and operation, and continues without cache

#### Scenario: LLM call failure catches LangChainException
- **WHEN** a tool's LLM call fails
- **THEN** the code catches `langchain_core.exceptions.LangChainException`, adds a note with the model and operation, and returns a `ToolResult.fail(...)` result

#### Scenario: Subprocess failure catches SubprocessError
- **WHEN** a subprocess spawned by a tool fails
- **THEN** the code catches `subprocess.SubprocessError`, adds a note with the command and return code, and returns a `ToolResult.fail(...)` result

#### Scenario: An unconfigured dependency is reported as unavailable
- **WHEN** a tool's optional dependency is not configured or not reachable
- **THEN** the tool returns the envelope's unavailable form carrying the reason, and does not report a generic failure

#### Scenario: A typed failure crossing into a tool keeps its code
- **WHEN** a tool calls a Result-typed service and receives a `Failure`
- **THEN** the envelope's error string carries that error's declared code, so the same failure is not described two different ways by two different tools

### Requirement: Degradation boundaries SHALL keep except Exception with add_note

The following locations SHALL keep `except Exception` because they are genuine degradation boundaries where:
1. The exception types are unknown at catch time (optional dependencies with opaque internals)
2. Too many exception types from multiple libraries could be thrown in a single block
3. The operation MUST succeed or degrade gracefully — crashing is never acceptable

Each of these sites SHALL add `exc.add_note()` with context and SHALL have a `# noqa: BLE001` comment with an explanatory reason.

A handler SHALL NOT be documented as catching every failure while its `except` clause names a closed list of exception types. Where a handler is deliberately partial, the requirement describing it SHALL say so and SHALL name what escapes.

#### Scenario: Optional dependency startup failure degrades gracefully
- **WHEN** an optional dependency (Neo4j, Graphiti, Crawl4AI, object storage) fails to initialize during app startup
- **THEN** the code catches `Exception`, adds a note with the dependency name and error details, sets the dependency to `None`, and continues startup

#### Scenario: LangGraph node failure returns fallback state
- **WHEN** a LangGraph node encounters an exception from an LLM, database, or external service
- **THEN** the code catches `Exception`, adds a note with the node name and operation, and returns a fallback state dict without crashing the graph

#### Scenario: OTEL instrumentation failure logs warning
- **WHEN** an OTEL instrumentor fails to register
- **THEN** the code catches `Exception`, adds a note with the instrumentor name, logs a warning, and continues without that instrumentor

#### Scenario: OTEL provider shutdown failure logs warning
- **WHEN** an OTEL provider fails to flush or shutdown
- **THEN** the code catches `Exception`, adds a note with the provider name, logs a warning, and continues shutdown

#### Scenario: The outbox scan and listen loops degrade on any failure
- **WHEN** the outbox relay's scan pass or its long-running listener raises for any reason
- **THEN** the code catches `Exception`, adds a note, logs, and skips or re-enters the loop, because a relay that dies leaves every pending event unpublished

#### Scenario: The outbox publish step is a partial handler, not a total one
- **WHEN** the outbox relay's publish step raises an exception outside the families it names
- **THEN** the event is **not** dead-lettered and the exception escapes to the surrounding loop, and this partiality is stated rather than described as dead-lettering on any failure

#### Scenario: Reranker model load failure falls back to default
- **WHEN** a CrossEncoder model fails to load (OSError, ValueError, RuntimeError from torch/transformers)
- **THEN** the code catches `Exception`, adds a note with the model name, loads the fallback model, and continues

#### Scenario: Reranker inference failure returns unranked results
- **WHEN** CrossEncoder reranking fails at inference time
- **THEN** the code catches `Exception`, adds a note with the model name and chunk count, returns chunks without reranking, and logs a warning

#### Scenario: Cognee recall failure returns empty list
- **WHEN** `cognee.recall()` fails for any reason (ad-hoc exceptions outside CogneeApiError hierarchy)
- **THEN** the code catches `Exception`, adds a note with the query and user_id, returns an empty list, and logs the failure

## ADDED Requirements

### Requirement: A repository SHALL NOT convert a caught third-party exception into a raise

A `try`/`except` around a third-party call inside a repository or service SHALL
classify the exception into a member of the feature's error union and return
`Failure`. It SHALL NOT re-raise, and SHALL NOT translate the exception into an
`APIException` subclass for the global handler to render.

This supersedes the retired bridge-and-raise pattern. Translation into an
exception type happens only at a boundary that must raise, and that boundary is
named in the layer classification, not in a repository.

#### Scenario: A caught library exception becomes a Failure
- **WHEN** a repository catches an exception from a third-party client
- **THEN** it returns `Failure` carrying a typed error, and the function has no `raise` statement in that handler

#### Scenario: The retired raise pattern is rejected
- **WHEN** code raises the result of translating a typed error into an exception from inside a repository or service
- **THEN** the project's enforcement rule reports a violation

#### Scenario: A raising boundary is still allowed to translate
- **WHEN** a FastAPI dependency or a WebSocket session receives a typed failure
- **THEN** it may translate and raise, because its layer is classified as exception-native

### Requirement: A custom exception family SHALL be rooted where its dispatcher can catch it

Every custom exception family SHALL be rooted so the handler responsible for it
catches it. A family rooted at a bare builtin — `RuntimeError`, `ValueError` — that
is expected to be rendered as an HTTP response or dead-lettered by a relay SHALL be
re-rooted under the project's own base, or its dispatcher SHALL name it explicitly.

Introducing a new exception family without naming the dispatcher that catches it
SHALL be a rule violation, because an uncaught family surfaces as an unhandled 500
or a silently abandoned queue row rather than as a diagnosable error.

#### Scenario: A RuntimeError-rooted family is unreachable today
- **WHEN** a circuit-breaker or agent-memory exception reaches the global exception handler
- **THEN** it is not matched by the handler's project-base branch and falls through to the unhandled path, and this change re-roots the family or widens the branch to name it

#### Scenario: A new family declares its dispatcher
- **WHEN** a new exception family is introduced
- **THEN** the handler that catches it is identified, and a family with no identified handler is a violation

#### Scenario: A module's own consumer catches everything the module raises
- **WHEN** a module raises both its own family and a framework exception from the same code path
- **THEN** its consuming handler catches both, so no raise site in that module is unhandled by its own caller
