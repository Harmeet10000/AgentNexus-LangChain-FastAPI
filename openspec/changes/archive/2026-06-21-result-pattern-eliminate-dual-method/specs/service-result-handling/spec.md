## ADDED Requirements

### Requirement: Service callers pattern-match AppResult

Every service or dependency that previously called a public wrapper method SHALL be updated to pattern-match on `AppResult[T]` instead.

#### Scenario: auth/service.py login (constant-time, uses `case _`)

```python
match await self._user_repo.find_by_email(dto.email):
    case Success(user) if user is not None and user.hashed_password is not None:
        resolved_user = user
    case _:
        verify_password(_DUMMY_HASH, dto.password)
        raise UnauthorizedException("Invalid credentials")
```

Note: Uses `case _` — no `log_expected_failure`, no error differentiation. This path must stay constant-time against timing attacks.

#### Scenario: auth/service.py verify_email (user-facing, logs failure)

```python
match await self._user_repo.find_by_verification_token_hash(hash_token(token)):
    case Success(user) if user is not None:
        ...
    case _:
        raise NotFoundException("Invalid or expired verification token")
```

#### Scenario: auth/service.py logout (write path, logs failure)

```python
match await self._token_repo.revoke_session(session_id=claims.jti, user_id=claims.sub, reason="logout"):
    case Success():
        pass
    case Failure(error):
        log_expected_failure(error, operation="logout")
        raise app_error_to_exception(error)
```

#### Scenario: auth/service.py oauth_callback (write path, logs failure)

```python
match await self._user_repo.find_or_create_oauth_user(...):
    case Success((user, created)):
        ...
    case Failure(error):
        log_expected_failure(error, operation="oauth_callback")
        raise app_error_to_exception(error)
```

#### Scenario: auth/dependencies.py get_current_user

```python
match await user_repo.find_by_id(claims.sub):
    case Success(user) if user is not None:
        return user
    case _:
        raise UnauthorizedException("User not found")
```

#### Scenario: documents/service.py get_status

```python
match await self.repo.fetch_status(user_id=user_id, document_id=document_id):
    case Success(record) if record is not None:
        warnings = _flatten_warnings(record.get("warnings", []))
        return DocumentStatusResponse(...)
    case _:
        raise NotFoundException("Document", document_id)
```

#### Scenario: search/service.py _run_parallel_search

```python
results = await asyncio.gather(
    repo.bm25_search(...),       # now returns AppResult
    repo.vector_search(...),     # now returns AppResult
    repo.trigram_search(...),    # now returns AppResult
    return_exceptions=False,
)
for r in results:
    match r:
        case Failure(error):
            raise app_error_to_exception(error)
return tuple(r.unwrap() for r in results)
```

### Requirement: Fix result.failure → result.failure() in LangGraph nodes

All occurrences of `result.failure` (property access) in `ingestion_kb/nodes.py` and `reconciliation/nodes.py` SHALL be changed to `result.failure()` (method call).

#### Scenario: ingestion_kb/nodes.py

- **WHEN** `result.failure` is called without parentheses
- **THEN** SHALL be `result.failure()` — `Failure.failure` is a method in this `returns` version, not a property

Affected lines in `ingestion_kb/nodes.py`: 98, 121, 160, 262, 308
Affected lines in `reconciliation/nodes.py`: 118, 179, 191, 258, 342

### Requirement: Add log_expected_failure to all Failure match branches

Every service `Failure(error)` match branch at the ownership boundary SHALL call `log_expected_failure(error, operation="...")` before raising or returning, EXCEPT constant-time paths in `auth/service.py` (login, forgot_password, resend_verification).

#### Scenario: auth/service.py refresh (already has it)

- **WHEN** `Failure(error)` is matched in `AuthService.refresh`
- **THEN** `log_expected_failure(error, operation="refresh_user_lookup")` SHALL be called before raising

#### Scenario: documents/service.py upload

- **WHEN** `Failure(error)` is matched after `create_document` returns `AppResult`
- **THEN** `log_expected_failure(error, operation="document_upload")` SHALL be called before raising

#### Scenario: search/service.py ingest

- **WHEN** `Failure(error)` is matched after `get_document_by_content_hash` returns `AppResult`
- **THEN** `log_expected_failure(error, operation="search_ingest")` SHALL be called before raising

### Requirement: SearchService unused imports cleanup

`SearchService.__init__` SHALL remove the `Failure` and `Success` imports from `returns.result` if they are no longer directly used.

#### Scenario: Clean imports

- **WHEN** `search/service.py` no longer directly constructs `Failure`/`Success`
- **THEN** the `from returns.result import Failure, Success` import SHALL be removed

### Requirement: Service pattern for write paths

For write operations (create, save, upsert, revoke), the service pattern SHALL be:
```python
match await repo.create(user):
    case Success(created):
        return created
    case Failure(error):
        log_expected_failure(error, operation="create_user")
        raise app_error_to_exception(error)
```

For idempotent writes (upsert, save):
```python
match await repo.save(user):
    case Success():
        pass
    case Failure(error):
        log_expected_failure(error, operation="save_user")
        raise app_error_to_exception(error)
```
