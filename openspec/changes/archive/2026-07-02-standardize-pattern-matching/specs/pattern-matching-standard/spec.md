## ADDED Requirements

### Requirement: Pattern matching taxonomy is documented
The project SHALL maintain a decision matrix in `.opencode/instructions/RESULT-PATTERN.md` that catalogs all pattern matching approaches used in the codebase and declares which to use in each scenario.

#### Scenario: Decision matrix exists
- **WHEN** a developer reads `.opencode/instructions/RESULT-PATTERN.md`
- **THEN** they find a table mapping scenarios (e.g., "Unwrapping AppResult[T]", "Routing on enum") to the correct pattern (e.g., "match/case on Result", "match/case on literals")

### Requirement: Service-layer Result unwrapping uses match/case
All service-layer code that unwraps `returns.result.Result` values SHALL use `match`/`case` with `Success`/`Failure` branches. The `isinstance(result, Failure)` pattern SHALL NOT be used in service-layer code.

#### Scenario: Auth service login unwraps user lookup
- **WHEN** `AuthService.login()` calls `self._user_repo.find_by_email(dto.email)`
- **THEN** the result is unwrapped with `match await ...: case Success(found) if ...: ... case Failure(error): ...`

#### Scenario: Document service upload unwraps document lookup
- **WHEN** `DocumentCommandService.upload_document()` calls `self.repo.get_document_by_user_hash(...)`
- **THEN** the result is unwrapped with `match await ...: case Success(existing) if ...: ... case Success(): ... case Failure(error): ...`

### Requirement: Enum and string literal dispatch uses match/case
Routing on closed sets of enum members or string literals SHALL use `match`/`case` instead of if/elif chains.

#### Scenario: Orchestrator routing
- **WHEN** `route_from_orchestrator()` receives a state with `orchestrator_action`
- **THEN** it dispatches with `match action.action_type: case OrchestratorActionType.START_PIPELINE: ...`

#### Scenario: OAuth provider config
- **WHEN** `get_oauth_config()` receives a provider string
- **THEN** it dispatches with `match provider: case "google": ... case "github": ... case _: raise ValidationException(...)`

### Requirement: Typed error hierarchy translation uses structural match/case
Translating between typed error hierarchies (e.g., `AppError` subtypes to `APIException` subtypes) SHALL use `match`/`case` with structural binding to extract fields.

#### Scenario: AppError to APIException mapping
- **WHEN** `app_error_to_exception()` receives a `NotFoundAppError(resource="User", identifier="123")`
- **THEN** it returns `NotFoundException(resource="User", identifier="123", error_code=...)` via structural binding

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
The pattern `x if isinstance(x, Model) else Model.model_validate(x)` SHALL be replaced with `match x: case Model(): ... case dict() as raw: Model.model_validate(raw)`.

#### Scenario: LangGraph state failure handling
- **WHEN** `features/ingestion/service.py` handles a failure value from LangGraph state
- **THEN** it uses `match failure: case AppError() as error: pass case dict() as raw: error = AppError.model_validate(raw)` instead of `isinstance(failure, AppError) else AppError.model_validate(failure)`
