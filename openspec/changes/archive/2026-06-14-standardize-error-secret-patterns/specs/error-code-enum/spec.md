## ADDED Requirements

### Requirement: ErrorCode StrEnum definition
The system SHALL define an `ErrorCode(StrEnum)` in `src/app/utils/error_codes.py` containing all error codes used across both `APIException` and `AppError` hierarchies.

#### Scenario: Enum values match existing string literals
- **WHEN** any error code is read from the enum
- **THEN** its value SHALL match the existing string literal it replaces (e.g., `ErrorCode.VALIDATION_ERROR == "VALIDATION_ERROR"`)

#### Scenario: Enum is importable from exceptions.py
- **WHEN** `from src.app.utils.exceptions import ErrorCode` is called
- **THEN** it SHALL resolve to the enum defined in `error_codes.py`

### Requirement: Exception classes reference ErrorCode enum
Every `APIException` subclass and `AppError` subclass SHALL use `ErrorCode` enum members as the default value for their `error_code`/`code` parameter, replacing string literals.

#### Scenario: ValidationException uses ErrorCode enum
- **WHEN** `ValidationException()` is instantiated
- **THEN** `error_code` SHALL default to `ErrorCode.VALIDATION_ERROR`

#### Scenario: NotFoundAppError uses ErrorCode enum
- **WHEN** `NotFoundAppError()` is instantiated
- **THEN** `code` SHALL default to `ErrorCode.NOT_FOUND`

#### Scenario: ExternalServiceException uses ErrorCode enum
- **WHEN** `ExternalServiceException(service="test", detail="msg")` is instantiated
- **THEN** `error_code` SHALL default to `ErrorCode.EXTERNAL_SERVICE_ERROR`

#### Scenario: Error codes in API responses are unchanged
- **WHEN** an exception is raised and caught by the global exception handler
- **THEN** the `error_code` value in the JSON response SHALL be identical to the pre-migration value
