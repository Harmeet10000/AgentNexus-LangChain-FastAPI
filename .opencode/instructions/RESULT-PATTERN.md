# Result / returns Pattern Rules

## When to Use `returns.Result`

**Yes — repositories and sync domain helpers:**
- Repository methods handling expected failures (not-found, conflict, DB error, validation). Dual-method pattern: `_result` variant returning `AppResult[T]` + thin public wrapper calling `app_error_to_exception(...)` on `Failure`.
- Validation, parsing, normalization, mapping functions where caller can make local decision.
- LangGraph node helpers returning typed `AppResult`, mapped to state-dicts at node boundary with `log_expected_failure()`.

**Yes — ownership boundaries (unwrapping `Failure`):**
- Service-layer: `match result: case Failure(error): raise app_error_to_exception(error)`.
- LangGraph node entrypoints: `match result: case Failure(error): log_expected_failure(...); return {...error_state...}`.

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
- Map internal `Failure(...)` to project exceptions before leaving the service layer.
- Match specific typed errors before generic ones; keep one final `Failure(error)` fallback at boundary.
- Do not pattern-match to swallow unexpected exceptions — unexpected failures should still raise.
- Use `FutureResult` only when async composition is materially clearer than ordinary async code.
- Import `Failure` and `Success` from `returns.result`.
