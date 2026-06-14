## ADDED Requirements

### Requirement: Secret fields use SecretStr
Any `Settings` field whose name contains `API_KEY`, `SECRET`, `PASSWORD`, or `TOKEN` (case-insensitive) SHALL use `pydantic.SecretStr` instead of `str`.

#### Scenario: Secret is redacted in repr()
- **WHEN** `repr(settings)` is called
- **THEN** the value of each migrated field SHALL display as `'**********'`

#### Scenario: Secret is redacted in model_dump()
- **WHEN** `settings.model_dump()` is called
- **THEN** the value of each migrated field SHALL display as `'**********'`

#### Scenario: Secret value is accessible via .get_secret_value()
- **WHEN** `settings.GEMINI_API_KEY.get_secret_value()` is called
- **THEN** the actual secret string SHALL be returned

#### Scenario: Non-secret fields remain plain str
- **WHEN** a non-secret field like `APP_NAME` or `HOST` is accessed
- **THEN** it SHALL remain typed as `str`

### Requirement: All downstream consumers call .get_secret_value()
Every site that reads a migrated field SHALL call `.get_secret_value()` to obtain the underlying `str`, ensuring no `str`-to-`SecretStr` type mismatch.

#### Scenario: LangChain client instantiation uses .get_secret_value()
- **WHEN** a LangChain client is initialized with an API key from settings
- **THEN** the call SHALL be `settings.GEMINI_API_KEY.get_secret_value()`

#### Scenario: SQLAlchemy connection string uses .get_secret_value()
- **WHEN** the `POSTGRES_PASSWORD` or `MONGODB_URI` secret is interpolated into a connection string
- **THEN** the call SHALL use `.get_secret_value()`
