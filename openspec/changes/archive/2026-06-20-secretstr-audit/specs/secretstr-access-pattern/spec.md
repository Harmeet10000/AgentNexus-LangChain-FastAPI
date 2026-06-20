## ADDED Requirements

### Requirement: SecretStr fields accessed via get_secret_value()
All Pydantic `SecretStr` fields SHALL be accessed via `.get_secret_value()`, never via `str()` coercion or f-string interpolation.

#### Scenario: Correct access pattern
- **WHEN** code needs the value of a `SecretStr` field
- **THEN** it SHALL call `field.get_secret_value()` and store the result in a local variable

#### Scenario: str() on SecretStr is flagged
- **WHEN** `str(settings.SECRET_FIELD)` or `f"{settings.SECRET_FIELD}"` is used on a `SecretStr` field
- **THEN** the CI check SHALL fail with a message indicating the correct pattern

### Requirement: Audit existing str() coercions
All 391 `str()` calls in `src/` SHALL be audited. Any that coerce a `SecretStr` object SHALL be replaced with `.get_secret_value()`.

#### Scenario: str() on SecretStr field is fixed
- **WHEN** a `str()` call wraps a field listed in `SECRET_FIELDS` or typed as `SecretStr`
- **THEN** it SHALL be replaced with `.get_secret_value()`

#### Scenario: str() on non-secret is left unchanged
- **WHEN** a `str()` call wraps a non-secret value
- **THEN** it SHALL NOT be modified

### Requirement: CI check prevents regression
A CI check SHALL exist that fails if `str()` is called on known secret field names.

#### Scenario: CI check runs on PR
- **WHEN** a pull request is opened or updated
- **THEN** the CI check SHALL scan `src/` for `str()` calls on secret fields and fail if any are found

#### Scenario: CI check uses field names from settings
- **WHEN** the CI check runs
- **THEN** it SHALL use the field names defined in `SECRET_FIELDS` (or equivalent) from `src/app/config/settings.py` as its list of protected names

### Requirement: Documentation updated
The project convention for `SecretStr` access SHALL be documented in `.opencode/instructions/PYTHON-TYPING-RULES.md`.

#### Scenario: Typing rules include SecretStr section
- **WHEN** a developer reads `PYTHON-TYPING-RULES.md`
- **THEN** there SHALL be a `### SecretStr` section explaining the `.get_secret_value()` convention and禁止 `str()` coercion
