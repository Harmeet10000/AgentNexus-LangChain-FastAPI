# Result / returns Pattern Rules

## Per-Feature Closed Union (ADR-001)

Each feature defines `errors.py` with `FeatureError` flat siblings, `Code` StrEnum, and `type FeatureErrorUnion = A | B | ...` closed with `assert_never` at every consumer. `kind`/`code`/`retryable` are `ClassVar`, never constructor args, and `ErrorKind` (7 members) is the only cross-feature vocabulary. No concrete inherits another concrete — a broader arm before a narrower one shadows silently.

## Pattern Matching Taxonomy

This codebase uses **3 pattern matching approaches**. This document catalogs each, declares the project standard, and marks which patterns are retired.

---

### Pattern 1: `isinstance` + `http_error()` / `render_result()` — **STANDARD**

The single canonical pattern for unwrapping `SubscriptionResult[T]` (per-feature `Result[T, SubscriptionError]`) in service-layer code. Expected failures are logged and answered with `http_error()` at service boundaries or `render_result(result, response, ...)` at routers — the typed error is NOT raised. `match`/`case` on `Success`/`Failure` is forbidden (ADR-002).

```python
result = await self._user_repo.find_by_email(dto.email)
if isinstance(result, Failure):
    error = result.failure()
    log_expected_failure(error, operation="find_by_email")
    return http_error(
        message=error.message,
        status_code=...,  # derive from error.kind (422 / 404 / 409 / 502 / 500-503)
        error_code=error.code,
        data=error.details,
    )
resolved = result.unwrap()
```

**Guard conditions** (null check after unwrap):
```python
result = await self._user_repo.find_by_id(user_id)
if isinstance(result, Failure):
    error = result.failure()
    log_expected_failure(error, operation="find_by_id")
    return http_error(
        message=error.message,
        status_code=...,  # derive from error.kind
        error_code=error.code,
        data=error.details,
    )
user = result.unwrap()
if user is None:
    raise NotFoundException("User not found")
```

**Return-value pattern** (when the success value is the return):
```python
result = await self._token_repo.get_user_sessions(user_id)
if isinstance(result, Failure):
    error = result.failure()
    log_expected_failure(error, operation="get_user_sessions")
    return http_error(
        message=error.message,
        status_code=...,  # derive from error.kind
        error_code=error.code,
        data=error.details,
    )
sessions = result.unwrap()
return [SessionResponse(...) for s in sessions]
```

**Used in:**
- `features/auth/service.py` — all Result unwrapping (register, login, logout, refresh, verify_email, forgot_password, reset_password, oauth_callback, list_sessions, revoke_session, etc.)
- `features/documents/service.py` — upload_document, get_status, search, ingestion, verification
- `features/search/service.py` — ingest_document, get_ingest_status, hybrid_search, rag_search
- `features/users/service.py` — _get_user_or_raise
- `features/auth/dependencies.py` — get_current_user
- `features/ingestion/service.py` — ingestion failure mapping

**When to use:** All service-layer and dependency code unwrapping repository/helper `AppResult[T]` values. This is the project standard. Do NOT `raise app_error_to_exception(error)` for expected failures — the legacy mapper-based raise is retired; `http_error()` is the only error formatter at this boundary.

---

### Pattern 2: `match`/`case` on enums and string literals — **STANDARD**

**~10 blocks.** Used for routing and dispatch on closed type sets.

```python
# Enum routing (agent_saul/nodes.py:254)
match action.action_type:
    case OrchestratorActionType.START_PIPELINE:
        return "ingestion"
    case OrchestratorActionType.CONTINUE:
        return action.target_node or "ingestion"
    case OrchestratorActionType.SYNTHESIZE:
        return "finalization"
    case OrchestratorActionType.DONE:
        return END
    case _:
        return "planner"

# String literal dispatch (auth/security.py:250)
match provider:
    case "google":
        return OAuthProviderConfig(...)
    case "github":
        return OAuthProviderConfig(...)
    case _:
        raise ValidationException(f"Unsupported OAuth provider: {provider}")
```

**When to use:** Dispatching on a closed set of enum members or string literals. Prefer over if/elif chains when the set is finite and the compiler can check exhaustiveness.

---

### Pattern 3: `match`/`case` on typed error hierarchy (structural matching) — **LEGACY**

**1 block, 7 cases** in `shared/result/mappers.py`. The boundary between the Result world and the exception world. Retained only for legacy call sites that still `raise app_error_to_exception(error)` — new code answers expected failures with `http_error()` (Pattern 1) and never raises the typed error.

```python
def app_error_to_exception(error: AppError) -> APIException:
    match error:
        case ValidationAppError():
            return ValidationException(
                detail=error.message, error_code=error.code, data=error.details,
            )
        case NotFoundAppError(resource=resource, identifier=identifier):
            return NotFoundException(
                resource=resource, identifier=identifier, error_code=error.code,
            )
        case ConflictAppError():
            return ConflictException(
                detail=error.message, error_code=error.code, data=error.details,
            )
        case ExternalServiceAppError(service=service):
            return ExternalServiceException(
                service=service, detail=error.message, error_code=error.code,
            )
        case InfrastructureAppError(retryable=True):
            return ServiceUnavailableException(
                detail=error.message, error_code=error.code, data=error.details,
            )
        case InfrastructureAppError():
            return DatabaseException(
                detail=error.message, error_code=error.code,
            )
        case AppError():
            return ValidationException(
                detail=error.message, error_code=error.code, data=error.details,
            )
```

**Key features:**
- Structural binding: `NotFoundAppError(resource=resource, identifier=identifier)` extracts fields directly
- Guard on `retryable=True` to distinguish retryable vs permanent infrastructure errors
- Most-specific to least-specific ordering (subclasses before `AppError` base)
- Catch-all `AppError()` at the end for any new error types added later

**When to use:** Translating between typed hierarchies. Only needed at system boundaries (Result→Exception, Exception→Response).

---

### Pattern 4: `isinstance` for exception dispatch — **KEEP (no change)**

**1 block, 4 branches** in `middleware/global_exception_handler.py`.

```python
if isinstance(exc, APIException):
    ...
elif isinstance(exc, RequestValidationError):
    ...
elif isinstance(exc, StarletteHTTPException):
    ...
# catch-all for unexpected errors
```

**Why not match/case:** The exception hierarchy is defined by FastAPI/Starlette (external), not by this project. The if/elif chain is idiomatic for exception handlers and reads clearly with the comment blocks separating each tier.

**When to use:** Exception handlers traversing external exception hierarchies. Do not convert to match/case — the if/elif chain is clearer here.

---

### Pattern 5: `isinstance` for defensive type guards on dynamic/untrusted data — **AUDIT AND REDUCE**

**~80+ occurrences across the codebase.** Two subcategories:

#### 5a. Legitimate dynamic data checks (KEEP)

These guard against genuinely unknown runtime types from external systems:

```python
# Redis returns bytes, str, or None depending on decoder config
raw = cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)

# LangChain messages — mixed types in a list, need filtering
system = [m for m in messages if isinstance(m, SystemMessage)]
without_tools = [m for m in messages if not isinstance(m, ToolMessage)]

# WebSocket inbound — discriminated union parsed by Pydantic
if isinstance(inbound, WSPingMessage):
    await self._send_json(ws, WSPongFrame().model_dump())
if not isinstance(inbound, WSResumeMessage):
    # error: expected resume
```

**Keep these.** They protect against genuinely dynamic data from Redis, LangChain, WebSocket, and Celery.

#### 5b. Redundant type checks on already-typed data (REMOVE)

These check types that are already guaranteed by the type system or Pydantic validation:

```python
# REDUNDANT — Pydantic model output is always a dict
if isinstance(exc.detail, dict)     # exc.detail is typed as dict | str
if isinstance(group, dict)          # iterating a list[dict]

# REDUNDANT — function signature guarantees the type
if isinstance(value, str)           # parameter annotated as str
if isinstance(value, list)          # parameter annotated as list

# REDUNDANT — Pydantic model field is typed
if isinstance(raw_groups, list)     # field: list[...] = ...
```

**Remove these.** They add noise without safety. If the type system says it's a `list[dict]`, trust it. If it might be `None`, use `if x is not None:` instead.

**When to use isinstance:** Only on data from external boundaries (Redis, HTTP responses, WebSocket frames, Celery results, LangChain internals) where the Python type system cannot guarantee the runtime type.

---

## Project Standard: Decision Matrix

| Scenario | Pattern | Example |
|---|---|---|
| Unwrapping `AppResult[T]` | `isinstance(result, Failure)` + `http_error()` | Pattern 1 |
| Routing on enum or closed string set | `match`/`case` on literals | Pattern 2 |
| Translating typed error hierarchy | `match`/`case` with structural binding | Pattern 3 |
| Exception handler (external hierarchy) | `isinstance` if/elif chain | Pattern 4 |
| Guarding data from Redis/LangChain/WS | `isinstance` type guard | Pattern 5a |
| Checking already-typed Python data | **REMOVE** — trust the type system | Pattern 5b |

---

## When to Use `returns.Result`

**Yes — repositories and sync domain helpers:**
- Repository methods handling expected failures (not-found, conflict, DB error, validation). Dual-method pattern: `_result` variant returning `AppResult[T]` + thin public wrapper handling the failure locally (return `None` or `http_error(...)`).
- Validation, parsing, normalization, mapping functions where caller can make local decision.
- LangGraph node helpers returning typed `AppResult`, mapped to state-dicts at node boundary with `log_expected_failure()`.

**Yes — ownership boundaries (unwrapping `Failure`):**
- Service-layer: `if isinstance(result, Failure): log_expected_failure(...); return http_error(...)` — do NOT raise the typed error.
- LangGraph node entrypoints: `if isinstance(result, Failure): log_expected_failure(...); return {...error_state...}`.

## When NOT to Use `returns.Result`

**No — transport, lifecycle, orchestration:**
- FastAPI routers, dependencies, middleware — raise project exceptions directly.
- Lifespan wiring — raise on failure.
- Celery task entrypoints — raise project exceptions directly.

**No — async service and LangGraph node signatures:**
- Keep async service methods as ordinary async functions raising project exceptions.
- Keep LangGraph node entrypoints as ordinary async functions returning plain dicts.

**No — transaction boundaries:**
- Inside DB transaction blocks, raise exceptions for rollback. Translate to `Failure()` only after the boundary.

**No — `None` is sufficient:**
- When "not found" is the only signal and caller doesn't need to know why, return `None`.

## Dual-Method Pattern

```python
# _result variant — returns AppResult[T]
async def find_by_email_result(self, email: str) -> AppResult[User | None]:
    try:
        user = await User.find_one(User.email == email.lower())
        if user is None:
            return Failure(NotFoundAppError(...))
        return Success(user)
    except PyMongoError as exc:
        return Failure(InfrastructureAppError(...))

# public wrapper — thin, re-raises on Failure
async def find_by_email(self, email: str) -> User | None:
    result = await self.find_by_email_result(email=email)
    if isinstance(result, Failure):
        return None
    return result.unwrap()
```

## General Rules

- `Failure.failure` is a **method** in this version of `returns`, not a property — always call `result.failure()`.
- Handle internal `Failure(...)` with `http_error()` before leaving the service layer — never raise the typed error.
- Match specific typed errors before generic ones; keep one final `Failure(error)` fallback at boundary.
- Do not pattern-match to swallow unexpected exceptions — unexpected failures should still raise.

## Anti-Patterns to Eliminate

1. **`isinstance` on Pydantic model outputs** — trust the type system. If the model says `list[dict]`, it's `list[dict]`.
2. **`match`/`case` on `returns.Result` Success/Failure** — use `isinstance(result, Failure)` + `http_error()` instead. Match/case on Result creates no-op branches (`case Success(): pass`) and doesn't narrow types for `ty`.
3. **`(_ for _ in ()).throw(...)` generator hack** — use `isinstance(result, Failure)` + `http_error()` directly. Lambda throw hacks are obscure and untestable.
4. **`log_expected_failure(e, ...) or app_error_to_exception(e)`** — the `or` relies on `log_expected_failure` returning `None` (side effect). Use two separate statements (`log_expected_failure(...)`, then `return http_error(...)`) instead.
5. **`if/elif` chains on error types** — use `match`/`case` with structural binding (Pattern 3) for typed error translation at system boundaries only.
