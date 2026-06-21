## ADDED Requirements

### Requirement: Delete public wrapper, rename _result to primary

For every dual-method pair in `auth/repository.py`, `users/repository.py`, `documents/repository.py`, and `search/repository.py`, the system SHALL delete the public wrapper method and rename the `_result` method to be the primary method (remove `_result` suffix).

#### Scenario: find_by_email in auth repository

```python
# Before
async def find_by_email(self, email: str) -> User | None:
    result = await self.find_by_email_result(email=email)
    if isinstance(result, Failure):
        return None
    return result.unwrap()

async def find_by_email_result(self, email: str) -> AppResult[User | None]:
    ...

# After
async def find_by_email(self, email: str) -> AppResult[User | None]:
    ...
```

#### Scenario: store_session in RefreshTokenRepository (was "raise" wrapper)

```python
# Before
async def store_session(self, session: SessionData) -> None:
    result = await self.store_session_result(session=session)
    if isinstance(result, Failure):
        raise app_error_to_exception(result.failure())
    return result.unwrap()

# After
async def store_session(self, session: SessionData) -> AppResult[None]:
    ...
```

### Requirement: Remove app_error_to_exception from repos

The system SHALL remove imports of `app_error_to_exception` from all 4 repository files.

#### Scenario: Imports cleaned

- **WHEN** refactored
- **THEN** `from app.shared.result import ... app_error_to_exception` SHALL be removed from `auth/repository.py`, `documents/repository.py`, `search/repository.py` (users already doesn't import it)

### Requirement: All repo methods return AppResult[T]

Every method in the affected repositories SHALL return `AppResult[T]` where `T` is the previous return type.

#### Scenario: Consistency

- **WHEN** a caller invokes `repo.find_by_email(email)` or `repo.create(user)`
- **THEN** the return type SHALL be `AppResult[User | None]` or `AppResult[User]`

### Requirement: Add _result variants to uncovered repo methods

Methods in `DocumentRepository` (`bm25_search`, `vector_search`, `trigram_search`) and `SearchRepository` (`create_document`, `upsert_chunks`, `fetch_chunks_by_ids`) that lack `_result` variants SHALL be wrapped with error handling.

#### Scenario: bm25_search in DocumentRepository gets error wrapping

- **WHEN** `DocumentRepository.bm25_search()` raises `SQLAlchemyError`
- **THEN** the error SHALL be captured as `Failure(InfrastructureAppError(...))` instead of crashing

#### Scenario: create_document in SearchRepository gets error wrapping

- **WHEN** `SearchRepository.create_document()` raises an integrity error
- **THEN** the error SHALL be captured as either `Failure(ConflictAppError(...))` or `Failure(InfrastructureAppError(...))` instead of crashing
