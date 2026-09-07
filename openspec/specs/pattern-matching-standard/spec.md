# Pattern Matching Standard Specification

## Purpose

Defines which pattern matching approach the project uses for each scenario: narrowing for Result containers, exhaustive match for closed error unions, and match for enum or literal dispatch.

## Requirements

### Requirement: Pattern matching taxonomy is documented
The project SHALL maintain a decision matrix in `.opencode/instructions/RESULT-PATTERN.md` that catalogs all pattern matching approaches used in the codebase and declares which to use in each scenario. The matrix SHALL distinguish the `Result` **container**, which is opened by narrowing, from the **error union** inside it, which is dispatched by `match`. Every duplicated copy of that document elsewhere in the repository SHALL state the same rule.

#### Scenario: Decision matrix exists
- **WHEN** a developer reads `.opencode/instructions/RESULT-PATTERN.md`
- **THEN** they find a table mapping scenarios (e.g., "Opening a `<Feature>Result[T]`", "Dispatching on a feature error union", "Routing on enum") to the correct pattern (e.g., "`isinstance(result, Failure)`", "`match` + `assert_never`", "match/case on literals")

#### Scenario: The matrix explains why, not only what
- **WHEN** a developer reads the row for opening a `Result`
- **THEN** it records that `match result: case Success(value)` does not narrow under the project's type checker, so the rule can be re-verified rather than taken on trust

#### Scenario: Duplicated copies do not drift
- **WHEN** a rule in `.opencode/instructions/RESULT-PATTERN.md` or `EXCEPTION-RULES.md` changes
- **THEN** every other copy of that document in the repository is updated in the same change, and a copy stating a superseded rule is a defect

### Requirement: Service-layer Result unwrapping uses isinstance narrowing

All code that unwraps a `returns.result.Result` SHALL use `isinstance(result, Failure)`. It SHALL NOT `match` on `Success`/`Failure`.

After `isinstance(result, Failure)` the type checker SHALL narrow `result.failure()` to the feature's error union and `result.unwrap()` to the success type. Code SHALL rely on that narrowing rather than re-asserting the type.

#### Scenario: Auth service login unwraps user lookup
- **WHEN** `AuthService.login()` calls `self._user_repo.find_by_email(dto.email)`
- **THEN** the result is opened with `if isinstance(result, Failure):`, and inside that branch `result.failure()` is typed as the users feature's error union

#### Scenario: Document service upload unwraps document lookup
- **WHEN** `DocumentCommandService.upload_document()` calls `self.repo.get_document_by_user_hash(...)`
- **THEN** the result is opened with `isinstance(result, Failure)`, and the not-found case is distinguished by matching the narrowed error rather than by a guard on the container

#### Scenario: The success value is narrowed too
- **WHEN** control passes the `isinstance(result, Failure)` branch without returning
- **THEN** `result.unwrap()` is typed as the success type, with no cast or assertion needed

#### Scenario: Matching the container is rejected by the gates
- **WHEN** code matches on `Success`/`Failure` in any form, including the bound forms `case Success(value):` and `case Failure(error):`
- **THEN** the project's enforcement rule reports a violation

### Requirement: Enum and string literal dispatch uses match/case
Routing on closed sets of enum members or string literals SHALL use `match`/`case` instead of if/elif chains.

#### Scenario: Orchestrator routing
- **WHEN** `route_from_orchestrator()` receives a state with `orchestrator_action`
- **THEN** it dispatches with `match action.action_type: case OrchestratorActionType.START_PIPELINE: ...`

#### Scenario: OAuth provider config
- **WHEN** `get_oauth_config()` receives a provider string
- **THEN** it dispatches with `match provider: case "google": ... case "github": ... case _: raise ValidationException(...)`

### Requirement: Typed error hierarchy translation uses structural match/case
Translating a typed error into a raising boundary's exception type SHALL use `match`/`case` with structural binding to extract fields, and SHALL close with `assert_never` where the source is a closed union.

Such translation SHALL occur only where a boundary must raise — a FastAPI dependency, a WebSocket session, a Celery task, an MCP handler. It SHALL NOT occur on the HTTP response path, where a failure is rendered from its `kind` rather than converted into an exception.

#### Scenario: AppError to APIException mapping
- **WHEN** the shared translation function receives an error carrying a resource and identifier and the calling layer must raise
- **THEN** it returns the corresponding exception with those fields extracted by structural binding, rather than by attribute access on a base type

#### Scenario: A raising boundary translates structurally
- **WHEN** an authentication dependency receives a typed failure it must reject the request for
- **THEN** it matches the error union with structural binding, raises the corresponding exception carrying the bound fields, and closes the match with `assert_never`

#### Scenario: The HTTP path does not translate
- **WHEN** an endpoint receives a typed failure
- **THEN** it renders the failure from its `kind` and does not convert it into an exception for the global handler

#### Scenario: A new error type breaks stale translations
- **WHEN** a member is added to a feature's error union
- **THEN** every structural translation over that union fails type-checking until an arm is added

### Requirement: External exception dispatch uses isinstance
Exception handlers that traverse external exception hierarchies (FastAPI, Starlette) SHALL use `isinstance` if/elif chains. These SHALL NOT be converted to match/case.

#### Scenario: Global exception handler
- **WHEN** `global_exception_handler()` receives an exception
- **THEN** it dispatches with `isinstance(exc, APIException)`, `isinstance(exc, RequestValidationError)`, `isinstance(exc, StarletteHTTPException)` in order

### Requirement: isinstance is preserved for genuinely dynamic data
isinstance checks on data from external boundaries (Redis, LangChain, WebSocket, Celery) SHALL be preserved and are considered correct.

#### Scenario: Redis bytes handling
- **WHEN** code reads from Redis and the value could be bytes or str
- **THEN** `isinstance(cached, bytes)` is used to determine decoding strategy

#### Scenario: LangChain message filtering
- **WHEN** code filters a mixed-type message list from LangChain
- **THEN** `isinstance(m, SystemMessage)` or `isinstance(m, ToolMessage)` is used for type-based filtering

### Requirement: Redundant isinstance checks on typed data are removed
isinstance checks on data whose type is already guaranteed by function annotations, Pydantic model fields, or other type system guarantees SHALL be removed.

#### Scenario: Pydantic model output check removed
- **WHEN** code previously checked `isinstance(exc.detail, dict)` where `exc.detail` is typed as `dict | str`
- **THEN** the isinstance check is removed and the code trusts the type annotation

#### Scenario: Typed parameter check removed
- **WHEN** code previously checked `isinstance(value, str)` where the parameter is annotated as `str`
- **THEN** the isinstance check is removed

### Requirement: isinstance+model_validate hybrid is replaced with match/case
Where a value may arrive either as a model instance or as its serialised mapping — a graph state field that has round-tripped through a checkpointer — the pattern `x if isinstance(x, Model) else Model.model_validate(x)` SHALL be replaced with `match x: case Model(): ... case dict() as raw: Model.model_validate(raw)`.

This is a genuinely dynamic boundary, not typed data. The `match` here is correct for the same reason the container `match` is not: the incoming value's type is unknown, so the class pattern is doing real work rather than failing to narrow an already-typed union.

#### Scenario: LangGraph state failure handling
- **WHEN** a service reads a failure value out of LangGraph state that may be a model instance or a mapping
- **THEN** it uses `match failure: case FeatureError() as error: pass case dict() as raw: error = <ConcreteError>.model_validate(raw)`, rather than an `isinstance`/`model_validate` conditional expression

#### Scenario: The deserialised value re-enters the closed union
- **WHEN** a mapping from graph state is validated back into an error instance
- **THEN** it is validated into a concrete member of the feature's union, so downstream exhaustive matches remain valid

### Requirement: Feature error unions are dispatched with exhaustive match/case

`match`/`case` SHALL be used on a feature's closed error union, after the `Result` has been opened by narrowing. Such a match SHALL close with `case _ as unreachable: assert_never(unreachable)`.

This is where `match` earns its place: over a closed union the type checker both narrows each arm and proves the set complete, so a new failure mode cannot be silently ignored. Over the `Result` container it does neither.

Where the scrutinee arrives as untyped graph-state data — a value that may be a model instance or its serialised mapping after a checkpointer round-trip — `isinstance` dispatch is permitted instead, per `.opencode/instructions/RESULT-PATTERN.md` Pattern 5a (defensive guards on dynamic/untrusted data). The SHALL above applies once the value is narrowed to the closed union. `features/ingestion/service.py:84-91` dispatches `FeatureError` vs `dict` at exactly such a boundary and is conforming without change.

#### Scenario: An exhaustive dispatch type-checks
- **WHEN** a service matches every member of its feature's error union and closes with `assert_never`
- **THEN** `uv run ty check src/` reports no diagnostic

#### Scenario: A missing arm fails the type check
- **WHEN** one member of the union has no arm
- **THEN** `uv run ty check src/` reports `type-assertion-failure` and names the uncovered member as the inferred argument type

#### Scenario: Arms are flat, never nested by inheritance
- **WHEN** a match dispatches on error types
- **THEN** no arm's type is a supertype of another arm's type, because a class pattern is `isinstance`-based and a broader arm would shadow a narrower one while still reporting the match exhaustive

#### Scenario: Untyped graph-state data dispatches with isinstance
- **WHEN** a service reads a failure value out of LangGraph state that may be a model instance or a mapping
- **THEN** it may dispatch with `isinstance` (RESULT-PATTERN.md Pattern 5a) rather than `match`, because the scrutinee is not yet narrowed to the closed union the SHALL above governs
