## Why

`AuthService.resend_verification()` and `AuthService.forgot_password()` independently duplicate the same 8-line engine-creation boilerplate to call `with_outbox`: they import `create_async_engine`, import `get_database_url`, create an engine, wrap it in an `AsyncSession`, call `with_outbox`, then dispose the engine. This pattern appears twice with identical structure and only differs in the event type and payload shape. Duplicating engine lifecycle management in a service layer is a code-quality regression — it normalizes throw-away database connections outside the canonical session factory lifecycle. If a third email-like outbox event emerges, this pattern will be copy-pasted a third time.

## What Changes

- **Extract `_publish_outbox_event()` as a private instance method on `AuthService`.** The method accepts `aggregate_type: str`, `aggregate_id: str`, `event_type: str`, `payload: dict[str, object]`, manages the engine lifecycle once, and delegates to `with_outbox`. Both `resend_verification` and `forgot_password` call it instead of duplicating the block.
- **No spec-level capability changes** — this is an internal extraction with no behavioral change. The method is private, so it does not appear in the service's public API.

### Alternate approach considered and rejected

We considered injecting `session_factory` from the PG connection pool via the lifespan, but `AuthService` currently operates on Beanie (MongoDB) with no PG dependency. Adding a `session_factory` to the constructor would push PG awareness into every auth route for the sake of two edge-case call sites. The private helper with a dedicated engine is simpler and keeps the PG concern contained. If a third caller appears, promote to a shared utility in `src/app/shared/outbox/`.

## Capabilities

### New Capabilities
*(none — internal refactoring only)*

### Modified Capabilities
*(none — no spec-level behavior changes)*

## Impact

- **File:** `src/app/features/auth/service.py` — add `_publish_outbox_event()` method; rewrite `resend_verification()` and `forgot_password()` to call it.
- **No new public API.** No new dependencies. No database migrations.
