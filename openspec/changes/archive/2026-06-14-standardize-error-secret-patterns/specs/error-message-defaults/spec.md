## ADDED Requirements

### Requirement: Dead message constants removed from enums.py
The system SHALL remove unused error message constants (`SOMETHING_WENT_WRONG`, `INTERNAL_SERVER_ERROR`, `VALIDATION_ERROR`, `NOT_FOUND`, `UNAUTHORIZED`, `FORBIDDEN`) from `src/app/config/enums.py`.

#### Scenario: Constants absent after migration
- **WHEN** `from src.app.config.enums import *` is inspected
- **THEN** the removed constants SHALL not be present
- **AND** no import in the codebase SHALL reference them

### Requirement: Message defaults consolidated in exceptions.py
Exception classes that previously used inline string defaults SHALL reference consolidated message constants or retain their existing contextual messages (e.g., `NotFoundException` uses `f"{resource} with ID '{identifier}' not found"`).

#### Scenario: ValidationException uses consolidated default
- **WHEN** `ValidationException()` is called with no arguments
- **THEN** `detail` SHALL default to `"Validation error"` (matching current behavior)

#### Scenario: Contextual messages remain unchanged
- **WHEN** `NotFoundException(resource="User", identifier=42)` is raised
- **THEN** the detail message SHALL remain `"User with ID '42' not found"`
