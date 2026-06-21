## ADDED Requirements

### Requirement: Service callers pattern-match AppResult

Every service or dependency that previously called a public wrapper method SHALL be updated to pattern-match on `AppResult[T]` instead.

#### Scenario: auth/service.py login (previously swallowed)

**Before:**
```python
user = await self._user_repo.find_by_email(dto.email)
if user is None or user.hashed_password is None:
    verify_password(_DUMMY_HASH, dto.password)
    raise UnauthorizedException("Invalid credentials")
```

**After:**
```python
match await self._user_repo.find_by_email(dto.email):
    case Success(user) if user is not None and user.hashed_password is not None:
        resolved_user = user
    case _:
        verify_password(_DUMMY_HASH, dto.password)
        raise UnauthorizedException("Invalid credentials")
```

#### Scenario: auth/service.py refresh (previously called _result directly)

**Before:**
```python
user_result = await self._user_repo.find_by_id_result(claims.sub)
match user_result:
    case Success(user) if user is not None:
        resolved_user = user
    case Success():
        raise UnauthorizedException("User not found or disabled")
    case Failure(error):
        log_expected_failure(error, operation="refresh_user_lookup")
        raise UnauthorizedException("Invalid token subject")
```

**After** (method renamed, no suffix change needed):
```python
match await self._user_repo.find_by_id(claims.sub):
    case Success(user) if user is not None:
        resolved_user = user
    case Success():
        raise UnauthorizedException("User not found or disabled")
    case Failure(error):
        log_expected_failure(error, operation="refresh_user_lookup")
        raise UnauthorizedException("Invalid token subject")
```

#### Scenario: auth/dependencies.py get_current_user

**Before:**
```python
user: User | None = await user_repo.find_by_id(claims.sub)
if user is None:
    raise UnauthorizedException("User not found")
```

**After:**
```python
match await user_repo.find_by_id(claims.sub):
    case Success(user) if user is not None:
        return user
    case _:
        raise UnauthorizedException("User not found")
```

#### Scenario: documents/service.py upload_document

**Before:**
```python
existing = await self.repo.get_document_by_user_hash(
    user_id=user_id, content_hash=content_hash
)
if existing is not None:
    return DocumentUploadResponse(..., duplicate=True)
```

**After:**
```python
match await self.repo.get_document_by_user_hash(
    user_id=user_id, content_hash=content_hash
):
    case Success(existing) if existing is not None:
        return DocumentUploadResponse(..., duplicate=True)
    case Success():
        pass  # no duplicate, continue
    case Failure(error):
        raise app_error_to_exception(error)
```

#### Scenario: documents/service.py get_status

**Before:**
```python
record = await self.repo.fetch_status(user_id=user_id, document_id=document_id)
if record is None:
    raise NotFoundException("Document", document_id)
```

**After:**
```python
match await self.repo.fetch_status(user_id=user_id, document_id=document_id):
    case Success(record) if record is not None:
        ...
    case _:
        raise NotFoundException("Document", document_id)
```

#### Scenario: search/service.py _run_parallel_search

**Before:**
```python
return tuple(await asyncio.gather(
    repo.bm25_search(...),       # raises on Failure
    repo.vector_search(...),     # raises on Failure
    repo.trigram_search(...),    # raises on Failure
))
```

**After (wraps each call in failure handling):**
```python
results = await asyncio.gather(
    repo.bm25_search(...),       # returns AppResult
    repo.vector_search(...),     # returns AppResult
    repo.trigram_search(...),    # returns AppResult
    return_exceptions=False,
)
for r in results:
    match r:
        case Failure(error):
            raise app_error_to_exception(error)
return tuple(r.unwrap() for r in results)
```
