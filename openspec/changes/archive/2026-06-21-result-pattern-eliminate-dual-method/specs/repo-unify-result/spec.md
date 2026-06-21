## ADDED Requirements

### Requirement: Delete public wrapper, rename _result to primary

For every dual-method pair in auth/repository.py, users/repository.py, documents/repository.py, and search/repository.py, the system SHALL delete the public wrapper method and rename the `_result` method to be the primary method (remove `_result` suffix).

#### Scenario: find_by_email in auth repository

**Before:**
```python
async def find_by_email(self, email: str) -> User | None:
    result = await self.find_by_email_result(email=email)
    if isinstance(result, Failure):
        return None
    return result.unwrap()

async def find_by_email_result(self, email: str) -> AppResult[User | None]:
    ...
    return Success(user) if user else Failure(NotFoundAppError(...))
```

**After:**
```python
async def find_by_email(self, email: str) -> AppResult[User | None]:
    ...
    return Success(user) if user else Failure(NotFoundAppError(...))
```

#### Scenario: create in auth repository (was a "raise" wrapper)

**Before:**
```python
async def create(self, user: User) -> User:
    result = await self.create_result(user=user)
    if isinstance(result, Failure):
        raise app_error_to_exception(result.failure())
    return result.unwrap()
```

**After:**
```python
async def create(self, user: User) -> AppResult[User]:
    ...
    return Success(created)
```

### Requirement: Remove app_error_to_exception from repos

The system SHALL remove imports of `app_error_to_exception` from repository files that no longer use it.

#### Scenario: auth/repository.py

- **WHEN** the refactor is complete
- **THEN** `from app.shared.result import ... app_error_to_exception` SHALL be removed (only `ConflictAppError`, `InfrastructureAppError`, `NotFoundAppError`, `ValidationAppError` remain)

#### Scenario: search/repository.py

- **WHEN** the refactor is complete
- **THEN** the import of `app_error_to_exception` SHALL be removed

### Requirement: All repo methods return AppResult[T]

Every method in the affected repositories SHALL return `AppResult[T]` where `T` is the previous return type.

#### Scenario: Consistency

- **WHEN** a caller invokes `repo.find_by_email(email)` or `repo.create(user)`
- **THEN** the return type SHALL be `AppResult[User | None]` or `AppResult[User]` — not `User | None` or `User` directly
