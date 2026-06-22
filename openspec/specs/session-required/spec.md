# session-required Specification

## Purpose
TBD - created by archiving change outbox-relay-session-refactor. Update Purpose after archive.
## Requirements
### Requirement: Session is a required parameter on all internal relay methods

The `_publish`, `_mark_published`, and `_mark_failed` methods on `OutboxRelay` SHALL accept `session: AsyncSession` as a required parameter (not `AsyncSession | None`).

#### Scenario: `_handle_notification` passes its session through to `_publish`

- **WHEN** `_handle_notification` receives a notification from the listen loop
- **AND** `_handle_notification` creates an `AsyncSession` from `self._session_factory`
- **THEN** `_handle_notification` SHALL pass this session to `_publish`
- **AND** `_publish` SHALL pass the session to both `_mark_published` and `_mark_failed`

#### Scenario: `run_startup_scan` creates a session before calling `_publish`

- **WHEN** `run_startup_scan` has events to publish during startup
- **THEN** `run_startup_scan` SHALL create a session via `self._session_factory()`
- **AND** pass that session to `_publish`
- **AND** NOT create its own `create_async_engine` separately

### Requirement: Retry limit extracted to a named constant

The maximum publish retry attempts SHALL be defined as a module-level `Final[int]` constant `_MAX_RETRIES` with value `5`.

#### Scenario: SQL queries use the constant value

- **WHEN** `run_startup_scan` or `_handle_notification` queries for unpublished events
- **THEN** the `WHERE` clause SHALL filter `publish_attempts < :max_retries` where `max_retries` equals `_MAX_RETRIES`

#### Scenario: Dead-letter threshold matches the constant

- **WHEN** `_mark_failed` increments `publish_attempts`
- **THEN** the dead-letter threshold comparison SHALL use `attempts >= _MAX_RETRIES`

### Requirement: Dead shutdown machinery removed or wired

The `OutboxRelay.shutdown()` method and `_running` flag SHALL be removed. The `lifespan.py` SHALL continue to use `asyncio.Task.cancel()` as the sole shutdown mechanism.

#### Scenario: `shutdown()` is removed from `OutboxRelay`

- **WHEN** `lifespan.py` shuts down the relay
- **THEN** it SHALL call `app.state.outbox_relay_task.cancel()` only
- **AND** NOT call `relay.shutdown()`

#### Scenario: `_running` field is removed

- **WHEN** `OutboxRelay.__init__` initializes
- **THEN** it SHALL NOT set `self._running`

