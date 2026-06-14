## ADDED Requirements

### Requirement: UserRepository unconverted methods gain Result variants
The `UserRepository` in `auth/repository.py` SHALL add `_result` variants for all methods currently lacking them: `find_by_email`, `find_by_verification_token_hash`, `find_by_reset_token_hash`, `create`, `save`, `email_exists`, `find_or_create_oauth_user`. Each variant SHALL return `AppResult[T]`. The existing public methods SHALL become thin wrappers calling the `_result` variant.

#### Scenario: find_by_email_result returns Success on found
- **WHEN** `find_by_email_result(email)` finds a user with matching email
- **THEN** it returns `Success(User(...))`

#### Scenario: find_by_email_result returns NotFoundAppError on missing
- **WHEN** `find_by_email_result(email)` finds no user
- **THEN** it returns `Failure(NotFoundAppError(...))`

#### Scenario: find_by_verification_token_hash_result returns NotFoundAppError
- **WHEN** no user has the given verification token hash
- **THEN** it returns `Failure(NotFoundAppError(...))`

#### Scenario: create_result returns ConflictAppError on duplicate
- **WHEN** `create_result(user)` encounters a MongoDB duplicate key error
- **THEN** it returns `Failure(ConflictAppError(...))`

#### Scenario: email_exists_result returns Success(bool)
- **WHEN** `email_exists_result(email)` runs successfully
- **THEN** it returns `Success(True)` or `Success(False)`

#### Scenario: find_or_create_oauth_user_result wraps errors appropriately
- **WHEN** the oauth user creation encounters a duplicate
- **THEN** it returns `Failure(ConflictAppError(...))`
- **WHEN** the oauth user lookup encounters an unexpected error
- **THEN** it returns `Failure(InfrastructureAppError(...))`

### Requirement: RefreshTokenRepository gains Result variants
The `RefreshTokenRepository` in `auth/repository.py` SHALL add `_result` variants for all its methods: `store_session`, `get_session`, `revoke_session`, `get_user_sessions`, `revoke_all_user_sessions`. Thin public wrappers SHALL preserve existing signatures.

#### Scenario: store_session_result returns InfrastructureAppError on Redis failure
- **WHEN** the Redis pipeline or MongoDB insert fails
- **THEN** it returns `Failure(InfrastructureAppError(...))`

#### Scenario: get_session_result returns Success on found
- **WHEN** a session exists for the given refresh token
- **THEN** it returns `Success(Session(...))`

#### Scenario: get_session_result returns Success(None) on cache miss
- **WHEN** no session exists for the given refresh token (Redis returns None)
- **THEN** it returns `Success(None)` — this is a valid business outcome, not an error

#### Scenario: get_session_result returns InfrastructureAppError on Redis failure
- **WHEN** the Redis `get` call raises an unexpected error
- **THEN** it returns `Failure(InfrastructureAppError(...))`

#### Scenario: revoke_session_result wraps errors
- **WHEN** the Redis pipeline or MongoDB update fails
- **THEN** it returns `Failure(InfrastructureAppError(...))`

#### Scenario: revoke_all_user_sessions_result wraps errors
- **WHEN** the Redis pipeline or MongoDB update fails
- **THEN** it returns `Failure(InfrastructureAppError(...))`

### Requirement: Auth service uses match/case for new Result variants
The `auth/service.py` SHALL add `match/case` unwrapping for any new `_result` variant calls that weren't previously covered.

#### Scenario: match/case at service boundary maps Failure to exception
- **WHEN** the service calls a `_result` variant and receives `Failure(NotFoundAppError(...))`
- **THEN** it maps to `NotFoundException` via `app_error_to_exception` and raises
- **WHEN** the service receives `Failure(InfrastructureAppError(...))`
- **THEN** it maps to `ServiceUnavailableException` and raises
