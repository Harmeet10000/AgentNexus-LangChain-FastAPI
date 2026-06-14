## ADDED Requirements

### Requirement: Documents repository has Result variants
The `documents/repository.py` DocumentRepository SHALL expose `_result` variants for all critical persistence methods that return typed `AppResult[T]` using `Success` / `Failure` with `AppError` subtypes. A thin public wrapper SHALL call the `_result` variant and unwrap `Failure` to `None` or raise the mapped exception.

#### Scenario: get_document_by_id returns Success on found
- **WHEN** `get_document_by_id_result` finds a document matching the ID
- **THEN** it returns `Success(Document(...))`

#### Scenario: get_document_by_id returns NotFoundAppError on missing
- **WHEN** `get_document_by_id_result` does not find a document matching the ID
- **THEN** it returns `Failure(NotFoundAppError(...))`

#### Scenario: get_document_by_id returns InfrastructureAppError on DB failure
- **WHEN** `get_document_by_id_result` encounters a `SQLAlchemyError` or database connectivity issue
- **THEN** it returns `Failure(InfrastructureAppError(...))`

#### Scenario: get_document_by_id wrapper returns Document on success
- **WHEN** `get_document_by_id` (public wrapper) receives `Success(doc)`
- **THEN** it returns `doc`

#### Scenario: get_document_by_id wrapper returns None on NotFound
- **WHEN** `get_document_by_id` (public wrapper) receives `Failure(NotFoundAppError(...))`
- **THEN** it returns `None`

#### Scenario: get_document_by_id wrapper raises on InfrastructureError
- **WHEN** `get_document_by_id` (public wrapper) receives `Failure(InfrastructureAppError(...))`
- **THEN** it maps to `ServiceUnavailableException` via `app_error_to_exception` and raises

#### Scenario: Create/upsert methods return ConflictAppError on constraint violation
- **WHEN** `create_document_result` or `upsert_chunks_result` hits a unique constraint or duplicate key error
- **THEN** it returns `Failure(ConflictAppError(...))`

#### Scenario: Create/upsert methods return InfrastructureAppError on DB failure
- **WHEN** `create_document_result` or `upsert_chunks_result` encounters an unexpected DB error
- **THEN** it returns `Failure(InfrastructureAppError(...))`

#### Scenario: fetch_status returns NotFoundAppError for non-existent document
- **WHEN** `fetch_status_result` cannot find a status row for the given document
- **THEN** it returns `Failure(NotFoundAppError(...))`

### Requirement: Search repository has Result variants
The `search/repository.py` SearchRepository SHALL follow the same pattern as DocumentRepository: `_result` variants returning `AppResult[T]` with thin public wrappers.

#### Scenario: get_document_by_content_hash returns Success on found
- **WHEN** `get_document_by_content_hash_result` finds a document matching the content hash
- **THEN** it returns `Success(Document(...))`

#### Scenario: get_document_by_content_hash returns NotFoundAppError on missing
- **WHEN** `get_document_by_content_hash_result` does not find a match
- **THEN** it returns `Failure(NotFoundAppError(...))`

#### Scenario: get_document_by_content_hash returns InfrastructureAppError on DB failure
- **WHEN** `get_document_by_content_hash_result` encounters a `SQLAlchemyError`
- **THEN** it returns `Failure(InfrastructureAppError(...))`

#### Scenario: Vector/BM25/trigram search methods wrap DB errors in InfrastructureAppError
- **WHEN** any search method (`bm25_search_result`, `vector_search_result`, `trigram_search_result`) encounters a `SQLAlchemyError`
- **THEN** it returns `Failure(InfrastructureAppError(...))`

#### Scenario: Search methods return Success with empty list on no results
- **WHEN** a search method finds no matching rows (not a DB error)
- **THEN** it returns `Success([])`

### Requirement: Existing callers continue to work unchanged
The public wrapper methods (non-`_result` suffixed) SHALL preserve their exact existing signatures and return types so no existing caller requires modification.

#### Scenario: Wrapper returns None for not-found
- **WHEN** an existing caller calls `repo.get_document_by_id(id)`
- **THEN** it receives `Document | None` as before, with `None` for not-found

#### Scenario: Wrapper raises for infrastructure errors
- **WHEN** an existing caller calls a wrapper method and the DB fails
- **THEN** `ServiceUnavailableException` is raised (mapped via `app_error_to_exception`)
