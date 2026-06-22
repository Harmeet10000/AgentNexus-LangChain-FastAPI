## ADDED Requirements

### Requirement: `AuthService` has a private outbox-publish helper

`AuthService` SHALL provide a private instance method `_publish_outbox_event(aggregate_type, aggregate_id, event_type, payload)` that manages the PostgreSQL engine lifecycle and delegates to `with_outbox`.

#### Scenario: `_publish_outbox_event` creates an engine, calls `with_outbox`, and disposes

- **WHEN** `_publish_outbox_event` is called
- **THEN** it SHALL call `create_async_engine(get_database_url())`
- **AND** create an `AsyncSession` bound to the engine
- **AND** call `with_outbox(session=..., aggregate_type=..., aggregate_id=..., event_type=..., payload=...)`
- **AND** call `engine.dispose()` in a `finally` block

#### Scenario: `resend_verification` uses the helper

- **WHEN** `resend_verification` needs to publish an outbox event
- **THEN** it SHALL call `self._publish_outbox_event(...)` with event_type `"auth.send_verification_email"`
- **AND** NOT create its own engine

#### Scenario: `forgot_password` uses the helper

- **WHEN** `forgot_password` needs to publish an outbox event
- **THEN** it SHALL call `self._publish_outbox_event(...)` with event_type `"auth.send_password_reset_email"`
- **AND** NOT create its own engine

## REMOVED Requirements

### Requirement: Duplicate engine lifecycle blocks in `resend_verification` and `forgot_password`

**Reason**: Consolidated into `_publish_outbox_event` helper
**Migration**: Both methods now call `self._publish_outbox_event(...)` instead of inlining `create_async_engine` / `dispose`
