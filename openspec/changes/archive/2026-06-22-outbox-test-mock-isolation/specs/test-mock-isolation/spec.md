## ADDED Requirements

### Requirement: Outbox tests use a minimal proxy module instead of blanket `AsyncMock()`

The test file `tests/unit/test_outbox.py` SHALL create a minimal `types.ModuleType("app.utils")` proxy that exposes only the `logger` symbol needed by `with_outbox`, instead of replacing the entire `app.utils` module with a single `AsyncMock()`.

#### Scenario: Logger mock is applied via minimal module proxy

- **WHEN** `TestWithOutbox` runs
- **THEN** `sys.modules["app.utils"]` SHALL contain a `types.ModuleType` instance with only `logger` set to `AsyncMock()`
- **AND** all other attributes on the proxy module SHALL remain undefined

#### Scenario: All existing assertions pass

- **WHEN** `test_inserts_row_and_notifies` executes
- **THEN** the test SHALL assert that `with_outbox` performs an INSERT followed by a pg_notify
- **AND** the test SHALL pass without importing `from unittest.mock import patch`

#### Scenario: Test handles rollback on exception

- **WHEN** `test_rollback_on_exception` executes
- **THEN** the test SHALL assert that a `RuntimeError` is raised when `pg_notify` fails
- **AND** the test SHALL pass with the same assertion logic
