# Exception Rules

## Hierarchy

```text
starlette.exceptions.HTTPException  (re-exported as fastapi.HTTPException)
  └── APIException                          ← base for all handled API errors
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
        ├── DatabaseException               → 500  (original_exc?)
        └── ExternalServiceException        → 502  (service, detail)
```

`APIException` **extends** `starlette.exceptions.HTTPException` (imported as `fastapi.HTTPException`). All subclasses are valid `HTTPException` instances: Starlette/FastAPI native handlers can catch them. The Global Exception Handler (`global_exception_handler.py`) intercepts them FIRST via `isinstance(exc, APIException)` to extract rich detail (`error_code`, `data`, structured `message`).

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

## Result bridge pattern

Repositories return `AppResult[T]` (`Result[T, AppError]` from `returns`). Service boundaries unwrap:

```python
result = await repo.find_by_email(email)
if isinstance(result, Failure):
    error = result.failure()
    log_expected_failure(error, operation="find_by_email")
    raise app_error_to_exception(error)
user = result.unwrap()
```

The mapper (`app_error_to_exception` in `shared/result/mappers.py`) converts `AppError` subtypes to `APIException` subclasses without losing error code or detail. See `RESULT-PATTERN.md` for the full dual-method pattern.

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
