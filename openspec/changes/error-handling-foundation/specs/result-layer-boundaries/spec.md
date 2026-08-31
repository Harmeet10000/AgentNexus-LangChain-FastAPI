## Purpose

Draws the line between code that returns typed failures and code that raises,
and states what every exception-native layer owes at the point it hands a failure
to domain code, so that no file in the repository is left without a rule.

## ADDED Requirements

### Requirement: The domain core SHALL be Result-typed regardless of transport

Repository and service methods SHALL return `<Feature>Result[T]`. They SHALL NOT
raise to signal an expected failure, and SHALL NOT vary their contract by which
transport eventually consumes them — a WebSocket handler calling a service SHALL
receive the same typed `Result` an HTTP router receives.

The same obligation applies to a module under `src/app/shared/` that classifies a
third-party exception on behalf of callers. Being cross-feature infrastructure
rather than a feature changes which union it owns, not whether it returns one; see
`shared-infrastructure-errors`.

Railway-oriented programming is a property of the domain core, not of the
transport.

#### Scenario: A service returns the same Result to every caller
- **WHEN** the same service method is called from an HTTP router and from a WebSocket session loop
- **THEN** both receive `<Feature>Result[T]`, and the service contains no branch that depends on the caller's transport

#### Scenario: An expected failure is returned, not raised
- **WHEN** a repository finds no row for a lookup that has a not-found member in its union
- **THEN** it returns `Failure(<Feature>NotFoundError(...))` and raises nothing

### Requirement: A Result SHALL be opened with isinstance, and a match SHALL be reserved for the error union

Code unwrapping a `Result` SHALL use `isinstance(result, Failure)`. It SHALL NOT
`match` on `Success`/`Failure`.

This is a measured constraint, not a preference. On this repository's `ty`,
`match result: case Success(value)` binds `value` to the union of the success and
error types — no narrowing occurs — while `isinstance(result, Failure)` followed
by `.failure()` narrows correctly to the feature's error union, and `.unwrap()`
narrows to the success type.

`match` SHALL be used on the error union obtained after that narrowing, where it
does narrow and where `assert_never` is meaningful.

#### Scenario: isinstance narrows both sides
- **WHEN** a service unwraps `SubscriptionResult[Subscription]` with `isinstance(result, Failure)`
- **THEN** `result.failure()` is typed as `SubscriptionError` and `result.unwrap()` is typed as `Subscription`

#### Scenario: Matching on the container is rejected
- **WHEN** code is written as `match result: case Success(value): ... case Failure(error): ...`
- **THEN** the enforcement rule reports a violation, because `ty` does not narrow through it and the bound values are the union of both sides

#### Scenario: The narrowed error is matched exhaustively
- **WHEN** a service needs different handling per failure mode
- **THEN** it opens the `Result` with `isinstance`, then matches the narrowed error union and closes with `assert_never`

### Requirement: try/except in domain code SHALL only wrap a call into code the project does not own

A `try`/`except` block inside a repository or service SHALL wrap only a call into
a third-party library — SQLAlchemy, httpx, redis-py, the Razorpay SDK, boto3, an
LLM provider client. It SHALL NOT wrap a call to another repository or service
method in this project, because those return a `Result` that is consumed by
narrowing, not by catching.

The `except` block SHALL classify the caught exception into a specific member of
the feature's union. Classification SHALL remain explicit and readable at each
site; it SHALL NOT be hidden behind a generic wrapper or context manager, because
which library exception maps to which typed error is feature-specific knowledge.

#### Scenario: A third-party call is wrapped
- **WHEN** a repository calls `session.flush()`
- **THEN** the call is inside a `try` whose `except IntegrityError` and `except SQLAlchemyError` blocks each classify into a named member of the feature's union

#### Scenario: An owned call is not wrapped
- **WHEN** a service calls its repository
- **THEN** the call is not inside a `try`, and the returned `Result` is opened with `isinstance`

#### Scenario: Classification stays at the site
- **WHEN** two repositories both catch `IntegrityError`
- **THEN** each maps it to its own feature's conflict error explicitly, and no shared helper performs the mapping on their behalf

### Requirement: An error constructed inside an except block SHALL be logged at construction

When a `FeatureError` is constructed inside an `except` block — something
actually threw — it SHALL be logged before the `Failure` is returned.

When a `FeatureError` is constructed from a plain check with no exception
involved — a `None` from `scalar_one_or_none()`, a failed state-transition guard —
it SHALL NOT be logged automatically. Some of these are ordinary control flow,
and logging every one turns normal branching into incident noise.

Where a mutation was flushed before the exception, the order SHALL be: classify,
roll back, log, return.

#### Scenario: An exception-derived failure is logged
- **WHEN** a repository catches `SQLAlchemyError` and constructs its infrastructure error
- **THEN** the error is logged inside the `except` block before `Failure` is returned

#### Scenario: A check-derived failure is not logged automatically
- **WHEN** a repository's `scalar_one_or_none()` returns `None` and a not-found error is constructed
- **THEN** no log line is emitted by the repository, and whether that not-found is worth logging is the caller's decision

#### Scenario: A not-found used as control flow stays silent
- **WHEN** a service uses a not-found result to decide that this is a first upload rather than a duplicate
- **THEN** the normal path emits no warning, because the failure is an expected branch and not an incident

### Requirement: Every exception-native layer SHALL be named, classified, and own its adapter

A layer that is exception-native in its own right SHALL stay exception-based, and
SHALL be listed with an adapter contract stating where it converts to or from a
typed `Result`.

Every file that **constructs, catches, propagates, or renders an error** SHALL
fall under exactly one classification. A file that does none of those — a DTO, a
schema, an ORM model, a settings module, an enum, a pure helper — is outside this
requirement and needs no classification; a rule for it would be a rule about
nothing. The classification SHALL be:

| Layer | Pattern | Adapter obligation |
|---|---|---|
| Repository, service (all features) | Result | is the domain core |
| Shared third-party wrappers (`shared/services/`, `shared/crawler/`) | Result | classify the library exception and own a union; see `shared-infrastructure-errors` |
| Provider adapters inside `shared/rag/` | Result at the provider boundary | the `_provider_failure` boundary classifies; the surrounding pipeline stays exception-native |
| Optional-dependency guards (`except ImportError`) | capability flag | not error handling; sets an availability flag and owes no classification |
| Pre-service policy guards in a router body (rate limit, quota) | exceptions | not error handling; a boolean policy answer, raised before the service is called |
| HTTP router | renders Result | see `http-result-rendering` |
| WebSocket session and transport loop | exceptions | converts service Results at the send boundary; close codes stay exceptions |
| `kb_retry` tenacity boundary | exceptions | raises on exhaustion; the calling node converts to a state failure |
| Razorpay client tenacity boundary | exceptions | the payments service converts its taxonomy into `PaymentError` |
| Celery task bodies (`src/tasks/`) | exceptions | framework owns retry and dead-letter; a task converts Results it consumes, and a broad catch names why the failure is survivable |
| Celery beat cron entries | exceptions | same as task bodies |
| Generation adapter behind a circuit breaker (`app/api/generation_with_cb.py`) | Result at the provider boundary | names the provider families it converts; SHALL NOT catch `Exception` and relabel it as service-unavailable |
| API version routers (`app/api/v1.py`, `v2.py`) | none | aggregate routers; construct, catch, propagate and render nothing, so no row's obligation applies |
| Framework-contract raises (Pydantic validators, PEP 562 module `__getattr__`, `NotImplementedError` stubs) | builtin raise | not error handling; the framework, not project code, reads the raise — exempt from the union rules |
| Seeders and data scripts (`src/database/seeders/`) | exceptions | operator-facing, no envelope; owes an accurate exit status and the identity of the step that failed |
| Example code (`app/examples/`) | whichever row its subject falls under | holds no exemption of its own; satisfies the same gates as production code, because its purpose is to be copied |
| FastAPI auth dependencies | exceptions | the only way to short-circuit a route; converts Results it consumes |
| LangGraph nodes | error in state | a node returns a state update and never raises |
| MCP tool handlers and middleware | dict envelope | FastMCP owns the protocol; converts Results it consumes |
| Circuit breaker and idempotency locks (`connections/celery_reliability.py`) | exceptions | must be reachable by a dispatcher, see below |
| Agent tool wrappers | protocol result | the tool protocol owns the shape |
| Global exception handler (`middleware/global_exception_handler.py`) | registration dispatch | owns the envelope; dispatches framework types by `isinstance`, exempt from the union rules |
| Request session dependency (`connections/postgres.py`) | exceptions | commits on clean exit, rolls back only on an escaping exception; never inspects a Result |
| Connection factories (`connections/neo4j.py`, `postgres.py`, `crawl4ai.py`) | exceptions | startup and pool construction; consumed by the lifespan's named handlers |
| Cache helpers (`utils/cache/`) | Result | classify a backend failure as a cache failure, not a database failure |
| Error vocabulary (`utils/exceptions.py`, `utils/codes.py`, `shared/result/errors.py`) | declares types | handles nothing; frozen for the migration, see `feature-error-contract` |
| Lifespan and unsupervised background tasks (`lifecycle/lifespan.py`) | exceptions | degradation boundary; names every family it survives, must log, never crash startup |
| HTTP middleware and health probes (`middleware/{server_middleware,health_check,otel,api_versioning}.py`) | exceptions or status dict | the global handler owns the envelope |
| SSE streaming | exceptions | post-flush failures have no envelope; must be logged |
| Scripts and CLI | exceptions | operator-facing, no envelope required |
| Alembic env and migration bodies | exceptions | out of the domain contract |

An exception-native layer SHALL NOT be converted to Result merely for uniformity.
It SHALL be converted only where the project owns the control flow.

A construct that resembles error handling but classifies nothing SHALL NOT be
brought into the contract, and SHALL NOT be reported as a violation by a gate.
Four kinds exist in this codebase and each is listed above: an `except ImportError`
that sets an availability flag; a pre-service policy guard; the dispatcher's
`isinstance` chain; and a raise that satisfies a framework contract. A policy guard is
the least obvious of the first three.
`features/crawler/router.py` calls `check_rate_limit`, which returns
`tuple[bool, dict]` — no `Result`, no exception caught — and raises
`TooManyRequestsException` at three sites when the answer is `False`. That raise is
a rejection of the request before any service runs, not the rendering of a failure a
service produced, so the rule that routers render rather than raise does not reach
it. `shared/services/rate_limiter.py`, which supplies the boolean, raises nothing
and catches nothing at all.

The fourth kind is distinguished by **who reads the raise**, not by the exception's
type. `app/config/settings.py:473` and `app/api/strict_envelope.py:26` raise
`ValueError` inside Pydantic validators, where `ValueError` *is* the signalling
protocol — Pydantic catches it and produces `ValidationError`, and returning a
`Result` there would make the field validate successfully. `src/database/__init__.py:37`
raises `AttributeError` from a PEP 562 module `__getattr__`, which `hasattr` and
`from … import` depend on. The same builtin raised in a service is read by project
code and converts like any other. `shared-infrastructure-errors` enumerates the three
sites and states the rule.

#### Scenario: Every error-handling file is classified
- **WHEN** a reviewer opens a file that constructs, catches, propagates, or renders an error
- **THEN** exactly one row of the classification applies to it, and such a file matching no row is a gap to be closed before the change is complete

#### Scenario: A file with no error handling needs no rule
- **WHEN** a reviewer opens a DTO, schema, ORM model, settings module, or pure helper that neither constructs, catches, propagates, nor renders an error
- **THEN** no row is expected to apply, and its absence from the classification is not a gap

#### Scenario: A non-feature directory is classified by role
- **WHEN** a reviewer opens a file under `connections/`, `lifecycle/`, `middleware/`, `shared/`, `utils/`, `app/api/`, `app/config/`, `app/examples/`, `src/database/` or `src/tasks/` that handles an error
- **THEN** a row applies to it by the role it plays — domain classifier, dispatcher, session dependency, degradation boundary, vocabulary, framework contract, task body, script or exemplar — and not being under `features/` is not grounds for exemption

#### Scenario: An excluded tree is not a coverage gap
- **WHEN** a reviewer finds unclassified error handling under `src/mcp_core/` or `src/lynk/`
- **THEN** it is not reported as a gap in this contract, because `src/mcp_core/` is excluded by the owner's decision and `src/lynk/` contains no Python — the exclusion is recorded as a non-goal rather than left as an implied omission

#### Scenario: A framework-contract raise is not a violation
- **WHEN** the gate encounters `raise ValueError` inside a Pydantic validator or `raise AttributeError` inside a module `__getattr__`
- **THEN** no violation is reported, because the framework reads that raise as its signalling protocol and a `Result` in its place would be read as success

#### Scenario: Example code holds no exemption of its own
- **WHEN** the gates are run over a completed change and a file under `app/examples/` demonstrates a forbidden pattern
- **THEN** it is reported and corrected on the same terms as production code, and no rule is given a path exclusion for that directory

#### Scenario: A policy guard in a router is not a rendering violation
- **WHEN** the gate encounters a router that raises a rate-limit or quota exception after a boolean check that returned `False`, before calling its service
- **THEN** no violation is reported, because no `Result` was produced and no exception was caught — the guard rejected the request rather than rendering a failure

#### Scenario: A session loop keeps raising
- **WHEN** a WebSocket session must close the connection on a revoked session
- **THEN** it raises its close-code exception rather than threading a `Result` up through the receive loop

#### Scenario: A retry boundary is the adapter
- **WHEN** a tenacity-wrapped call exhausts its attempts and raises
- **THEN** the calling node or service catches it once and converts it into a typed failure, and the retry machinery itself is not reimplemented around Results

#### Scenario: A layer that consumes a Result converts at its own edge
- **WHEN** a Celery task, an auth dependency, or an MCP handler calls a service and receives a `Failure`
- **THEN** it converts that failure into its own mechanism — a raise, a dict envelope, or a retry — at that call site

### Requirement: An error family SHALL be reachable by the dispatcher that is expected to handle it

Every custom exception family SHALL be rooted so that the dispatcher responsible
for it can catch it. A family rooted at `RuntimeError` that is expected to be
rendered as an HTTP response or dead-lettered by a Celery relay SHALL be
re-rooted, or the dispatcher SHALL be widened to name it explicitly.

This closes a live gap. Six families under `connections/` and `shared/` are rooted
at `RuntimeError`, bare `Exception` or `ValueError`, and **five of the six are caught
nowhere in the repository** — `CircuitBreakerOpenError`, `IdempotencyLockError`,
`AgentMemoryError`, `CogneeSetupError` and `StateSchemaVersionError` each have raise
sites and zero catch sites. None is reachable by the global exception handler's
`APIException` branch, and none by the outbox relay's publish handler, which names
only `CeleryError` and `PostgresError`. The one counter-example,
`TransientExternalError`, is caught by name at four consuming nodes and is the shape
to copy. The WebSocket router likewise catches only its own violation family while
Starlette `WebSocketException` is raised alongside it from the same module. The full
inventory and the per-directory obligations are in `shared-infrastructure-errors`.

A dead-letter path SHALL either catch every family its publish step can raise, or
be documented as partial. It SHALL NOT be described as total while naming a closed
list of exception types.

#### Scenario: An HTTP-facing family reaches the envelope
- **WHEN** a request path raises a member of the circuit-breaker family
- **THEN** the global exception handler renders it into the standard error envelope with a deliberate status, rather than falling through to the unhandled 500 branch

#### Scenario: A task-facing family reaches the dead-letter path
- **WHEN** a Celery-dispatched operation raises a member of a `RuntimeError`-rooted family during an outbox publish
- **THEN** the publish handler's catch names that family, so the event is dead-lettered rather than escaping the handler and leaving the row unclaimed

#### Scenario: A partial dead-letter path is described as partial
- **WHEN** a dead-letter handler catches a closed list of exception types
- **THEN** the requirement describing it does not claim it dead-letters on any failure, and the families it cannot catch are named

#### Scenario: A handler catches every family its module raises
- **WHEN** a module raises both its own violation family and a framework exception from the same code path
- **THEN** the consuming handler catches both, and no raise site in that module is left unhandled by its own consumer

### Requirement: Every enforcement rule SHALL be verified against both the form it forbids and the form it permits

An enforcement rule introduced or changed by this work SHALL be accompanied by a
fixture containing at least one example of the construct it forbids and one
example of the nearest construct it permits. The rule SHALL be shown to flag the
first and not the second before any violation count derived from it is relied on.

A rule's message SHALL describe what the rule actually matches. Where a rule's
coverage is narrower than its message, either the pattern or the message SHALL be
corrected.

This is not ceremony. The entire design rests on gates: a closed union is closed
only because a rule says nothing may subclass the base elsewhere, and a flat
sibling set is flat only because a rule says so — there is no type-checker
backstop for either. An unverified gate is indistinguishable from no gate, while
reading as coverage.

The `no-match-on-result` rule demonstrates the failure. Its pattern matches only
the argument-less `case Success()` / `case Failure()` form, so
`case Success(value):` — the exact construct the superseded spec mandated and this
work forbids — passes it unflagged, while its message asserts that match/case does
not narrow `Result`. Its reported count of zero violations is therefore not
evidence that the codebase is clean.

#### Scenario: A rule is shown to flag what it forbids
- **WHEN** the rewritten container-match rule is run against a fixture containing `case Success(value):` and `case Failure(error):`
- **THEN** it reports a violation for each

#### Scenario: A rule is shown not to flag what it permits
- **WHEN** the same rule is run against a fixture containing an exhaustive `match` over a feature's error union with concrete-type arms and an `assert_never` close
- **THEN** it reports no violation, because that construct is the one the design requires

#### Scenario: A count from a corrected rule is re-measured, not carried forward
- **WHEN** a rule's pattern is corrected so that its coverage widens
- **THEN** the violation count is re-measured with the corrected rule, and the previous count is not cited as a baseline

#### Scenario: A rule whose message overstates its coverage is corrected
- **WHEN** a rule's message describes a construct its pattern cannot match
- **THEN** either the pattern is widened to match it or the message is narrowed to what it does match, and the discrepancy is not left in place
