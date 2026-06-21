## Context

Every repository (auth, users, documents, search) implements a dual-method pattern:

```python
# Public wrapper — varies per method
async def find_by_email(self, email: str) -> User | None:
    result = await self.find_by_email_result(email=email)
    if isinstance(result, Failure):
        return None  # swallows
    return result.unwrap()

# _result variant
async def find_by_email_result(self, email: str) -> AppResult[User | None]:
    try:
        user = await User.find_one(User.email == email.lower())
        ...
    except PyMongoError as exc:
        return Failure(InfrastructureAppError(...))
```

The public wrapper has **two inconsistent behaviors**:
- **Swallow** (queries): return `None`/`[]` on any `Failure`
- **Raise** (writes): `raise app_error_to_exception(result.failure())`

Callers using the public wrapper (e.g., `auth/service.py`) never see `Failure` — the error is either swallowed or raised. This means infrastructure errors (DB down, Redis timeout) are silently converted to "not found" on query paths.

## Goals / Non-Goals

**Goals:**
- All repository methods return `AppResult[T]` — single method per operation
- All service/dependency callers pattern-match on `AppResult`
- Infrastructure failures propagate instead of being silently swallowed
- Remove `_result` suffix and `app_error_to_exception` from repos

**Non-Goals:**
- Add Result pattern to methods that don't already have it (create_document in search_repo, upsert_chunks in search_repo — they're simple enough)
- Change the `AppResult` type or `AppError` hierarchy
- Change `returns.result` library version

## Decisions

### D1: Drop suffix, drop wrapper, one method per operation

**Choice**: For every dual-method pair, delete the public wrapper and rename `_result` → primary name.

**Before:**
```python
async def find_by_email(self, email: str) -> User | None:
    result = await self.find_by_email_result(email=email)
    if isinstance(result, Failure):
        return None
    return result.unwrap()

async def find_by_email_result(self, email: str) -> AppResult[User | None]:
    try:
        user = await User.find_one(...)
        return Success(user) if user else Failure(NotFoundAppError(...))
    except PyMongoError as exc:
        return Failure(InfrastructureAppError(...))
```

**After:**
```python
async def find_by_email(self, email: str) -> AppResult[User | None]:
    try:
        user = await User.find_one(...)
        return Success(user) if user else Failure(NotFoundAppError(...))
    except PyMongoError as exc:
        return Failure(InfrastructureAppError(...))
```

**Rationale**: One method per operation. The `_result` suffix was a patch for the two-method problem. With one method, it's redundant.

**Alternative considered**: Keep both but make public wrappers always raise — rejected because it still has 2x code for no benefit.

### D2: Service layer pattern-matches `Failure` with explicit error discrimination

**Choice**: Service callers match on `Failure` and distinguish `NotFoundAppError` from `InfrastructureAppError`.

**Before:**
```python
# auth/service.py
user = await self._user_repo.find_by_email(dto.email)
if user is None:
    verify_password(_DUMMY_HASH, dto.password)
    raise UnauthorizedException("Invalid credentials")
```

**After:**
```python
match await self._user_repo.find_by_email(dto.email):
    case Success(user) if user is not None:
        resolved_user = user
    case _:
        verify_password(_DUMMY_HASH, dto.password)
        raise UnauthorizedException("Invalid credentials")
```

Note: The `_` wildcard in service-layer login/reset flows is intentional — these paths must stay constant-time regardless of whether the email exists OR the DB is down. For admin/user-facing operations, infrastructure errors should propagate instead.

**Rationale**: The match forces the caller to consider "what if the DB is down?" instead of silently getting `None` and crashing later.

### D3: Remove `app_error_to_exception` from repository imports

**Choice**: `app_error_to_exception` is no longer needed in repo files — it was only used by the "raise" public wrappers.

**Before:**
```python
from app.shared.result import ConflictAppError, InfrastructureAppError, NotFoundAppError, app_error_to_exception
```

**After:**
```python
from app.shared.result import ConflictAppError, InfrastructureAppError, NotFoundAppError
```

## Risks / Trade-offs

- **Risk**: Callers that previously got `None` from a swallowed `Failure` now see `Failure(InfrastructureAppError(...))` which they must handle. → **Mitigation**: Each caller gets a match statement. Infrastructure errors become `ServiceUnavailableException` or `DatabaseException` instead of silent swallow.
- **Risk**: `auth/service.py` login/reset-password/forgot-password must remain constant-time vs timing attacks. → **Mitigation**: These use `case _` wildcard that handles both `Failure` and `Success(None)` identically.
- **Risk**: `users/service.py` already calls `find_by_id_result` and pattern-matches — no change needed there. → Already verified.

## Migration Plan

1. Refactor `auth/repository.py` — delete 13 wrappers, rename 13 `_result` methods
2. Refactor `users/repository.py` — delete 1 wrapper, rename 1 `_result` method
3. Refactor `documents/repository.py` — delete 5 wrappers, rename 5 `_result` methods
4. Refactor `search/repository.py` — delete 5 `_result` wrappers, rename 5 methods (bm25/vector/trigram + get_document_*)
5. Update `auth/service.py` — 7 callers: login, refresh, verify_email, resend_verification, forgot_password, reset_password, logout, oauth_callback
6. Update `auth/dependencies.py` — 1 caller: get_current_user
7. Update `documents/service.py` — 5 callers: upload_document (get_document_by_user_hash, create_document), get_status (fetch_status), process_document_ingestion (upsert_chunks), _verify_legal_chunks (upsert_chunks)
8. Update `search/service.py` — 5 callers: ingest_document, get_ingest_status, parallel search, upsert_chunks
