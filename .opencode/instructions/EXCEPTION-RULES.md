# Exception Rules

## Per-Feature Typed Errors (ADR-001, D7)

Each feature owns its `errors.py` with a closed union:

```python
# src/app/features/subscriptions/errors.py
class SubscriptionCode(StrEnum): ...  # feature's own codes, never bare str
class SubscriptionNotFoundError(FeatureError):  # flat sibling, inherits FeatureError directly
    kind: ClassVar[ErrorKind] = ErrorKind.NOT_FOUND
    code: ClassVar[SubscriptionCode] = SubscriptionCode.SUBSCRIPTION_NOT_FOUND
type SubscriptionError = SubscriptionNotFoundError | SubscriptionDuplicateError | ...
type SubscriptionResult[T] = Result[T, SubscriptionError]
```

- `ErrorKind` (7 members) is the only cross-feature vocabulary; `kind` is `ClassVar[ErrorKind]` and never appears in `model_dump`.
- `code` is `ClassVar[FeatureCode StrEnum]` — hand-written `"STRING"` is rejected by `ty` (`invalid-assignment`).
- Concrete types are **flat siblings** — no concrete inherits another concrete. A broader arm before a narrower one shadows silently; `ty` still reports exhaustive via `assert_never`, so the ordering footgun is invisible.
- Where two features exchange a failure, the caller translates into its own union — no cross-feature error imports.

## Hierarchy (APIException family — transport, not Result)
        ├── ValidationException             → 422
        ├── NotFoundException               → 404  (resource, identifier?)
        ├── UnauthorizedException           → 401  (auto-adds WWW-Authenticate)
        │    ├── InvalidTokenException
        │    ├── ExpiredTokenException
        │    └── InvalidRefreshTokenException
        ├── ForbiddenException              → 403
        ├── ConflictException               → 409  (detail, data?)
        ├── TooManyRequestsException        → 429
        ├── ServiceUnavailableException     → 503
        ├── InfrastructureException         → 500/503  (retryable flag picks the status)
        ├── DatabaseException               → 500  (original_exc?)
        └── ExternalServiceException        → 502  (service, detail)
```

`APIException` **extends** `starlette.exceptions.HTTPException` (imported as `fastapi.HTTPException`). All subclasses are valid `HTTPException` instances: Starlette/FastAPI native handlers can catch them. The Global Exception Handler (`global_exception_handler.py`) intercepts them FIRST via `isinstance(exc, APIException)` to extract rich detail (`error_code`, `data`, structured `message`), then formats the envelope with `http_error()`.

`AppError` (typed Result failures) is **never** handled by the Global Exception Handler — it is a Pydantic model, not an `Exception`, and is never raised. It exists only inside `AppResult[T]` return values at repository/adapter boundaries. Expected `AppError` failures are answered with `http_error()` at the service-layer ownership boundary — they are NOT raised (see "Result bridge pattern" below). The GEH's only formatter is `http_error()`.

## Raise — let GEH handle it

In routers, services, dependencies, middleware, Celery task entrypoints — raise typed exceptions directly. Do NOT format HTTP responses manually.

```python
# ✅ Correct — typed, conveys intent, GEH formats the response
raise NotFoundException("User", user_id)
raise ValidationException("Invalid email format", data={"field": "email"})
raise ConflictException("Email already registered")
raise ServiceUnavailableException("Database connection pool exhausted")

# ❌ Wrong — use typed exceptions, not raw HTTPException
raise HTTPException(status_code=404, detail="User not found")

# ❌ Wrong — don't format error responses yourself
return JSONResponse(status_code=422, content={"error": ...})
```

The Global Exception Handler (`global_exception_handler`) dispatches via `isinstance`:
1. `APIException` → extracts `error_code`, `data`, structured `message` → uniform `APIResponse` envelope
2. `RequestValidationError` → 422 with per-field error array
3. `StarletteHTTPException` (plain) → generic `HTTP_{status_code}` error code
4. Anything else → 500 `INTERNAL_SERVER_ERROR` with traceback in non-prod

## Catch — explicit and narrow

Catch specific exception types. Avoid bare `except Exception`. When you must catch broad (e.g., at WebSocket or outer loop boundaries), use `except Exception as e:` and log with `log.exception(...)`.

```python
# ✅ Correct — narrow catch
try:
    result = await repository.find_by_id(id_)
except NotFoundException:
    return None

# ✅ Correct — catch + wrap with add_note (Python 3.11+)
try:
    await external_api.call()
except ExternalServiceException as e:
    e.add_note(f"Failed for user {user_id} during onboarding flow")
    raise

# ❌ Wrong — bare except swallows signals
try:
    ...
except Exception:
    pass

# ❌ Wrong — catching and re-raising as plain HTTPException loses context
try:
    await service.call()
except ServiceUnavailableException:
    raise HTTPException(503, "down")  # loses error_code, structured detail
```

### `e.add_note()` pattern

Use `e.add_note(...)` on an exception before re-raising to attach contextual info without losing the original traceback. The note appears in the exception chain traceback output and is visible to logging/tooling that inspects `__notes__` (Python 3.11+).

```python
try:
    await save_to_cognee(doc)
except DatabaseException as e:
    e.add_note(f"doc_id={doc.doc_id}, correlation_id={state.correlation_id}")
    e.add_note("Cognee write skipped — continuing with partial state")
    raise  # GEH catches the re-raised DatabaseException
```

Use `add_note` when:
- The exception crosses a layer boundary and the caller needs context
- You caught an exception to add info but can't handle it (must re-raise)
- You want to attach runtime state (IDs, parameters) for debugging

Do NOT use `add_note` when:
- You can handle the exception (return fallback, suppress it)
- The context is already in structured log fields (`logger.bind(...)`)
- You're about to raise a DIFFERENT exception type — use `raise ... from e` instead

### `try`/`except` as third-party adapter only

`try`/`except` in this codebase is an **adapter at the boundary of code you do not own** — `sqlalchemy`, `pymongo`, `redis`, `httpx`, `neo4j` etc. — where you catch the library's exception, classify it (`ErrorKind` + `code`), roll back if needed, and return a typed `Failure`. Application code never raises to communicate an expected failure to itself; it returns `Result`. Raising is for transport (router/dependency/middleware/Celery) or for truly unexpected faults that must reach the global handler.

## Result bridge pattern (per-feature `Result`)

Repositories return `SubscriptionResult[T]` (`Result[T, SubscriptionError]` — closed union per feature; `AppResult` is legacy and frozen). Service boundaries unwrap with `isinstance(result, Failure)` and **never** `match` on `Success`/`Failure` (ADR-002). Expected failures are answered, not raised — `render_result(result, response, ...)` derives the transport status from `error.kind` via `STATUS_BY_KIND` and emits the `http_error` envelope. `http_error()` remains the only formatter; `raise app_error_to_exception` is retired.

```python
# inside service or repository helper returning Result — translate
result = await repo.find_by_email(email)
if isinstance(result, Failure):
    error = result.failure()
    log_expected_failure(error, operation="find_by_email")
    return http_error(  # or render_result at router
        message=error.message,
        status_code=...,  # derive from error.kind (422 / 404 / 409 / 502 / 500-503)
        error_code=error.code,
        data=error.details,
    )
user = result.unwrap()
```

```python
result = await repo.find_by_email(email)
if isinstance(result, Failure):
    error = result.failure()
    log_expected_failure(error, operation="find_by_email")
    return http_error(
        message=error.message,
        status_code=...,  # derive from error.kind (422 / 404 / 409 / 502 / 500-503)
        error_code=error.code,
        data=error.details,
    )
user = result.unwrap()
```

Raising the typed error (`raise app_error_to_exception(error)`) is removed from the pattern — `http_error()` is the only error formatter at this boundary. The mapper (`app_error_to_exception` in `shared/result/mappers.py`) still exists for legacy call sites; new code does not raise it. See `RESULT-PATTERN.md` for the full dual-method pattern.

## LangGraph nodes

LangGraph nodes do NOT raise exceptions for expected failures. Return error state dicts instead:

```python
try:
    result = await tool.ainvoke(...)
except ExternalServiceException as e:
    e.add_note(f"step={step.step_id}")
    return {"errors": [AgentError(node="deep_research", code="TOOL_FAILED", ...)]}
```

Raising from a node will crash the graph branch. Only raise for truly unrecoverable infrastructure errors.

## WebSocket exceptions

Use `WebSocketException` from FastAPI for WebSocket-specific failures:

```python
from fastapi import WebSocketException, status

raise WebSocketException(
    code=status.WS_1008_POLICY_VIOLATION,
    reason="Session expired or revoked",
)
```

The WebSocket router catches `WebSocketDisconnect`, `ValidationException`, and `Exception` in layered handlers, mapping each to the appropriate close code or error frame.
