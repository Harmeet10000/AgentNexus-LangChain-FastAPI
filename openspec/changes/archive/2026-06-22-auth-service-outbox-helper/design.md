## Context

`AuthService` in `src/app/features/auth/service.py` uses Beanie (MongoDB) for persistence and has no PostgreSQL session dependency. Two methods — `resend_verification` and `forgot_password` — need to write an outbox event after mutating the user record in MongoDB. Both methods independently create a throw-away async engine via `create_async_engine(get_database_url())`, wrap it in an `AsyncSession`, call `with_outbox`, then `await engine.dispose()`. The block is duplicated verbatim across both methods (8 lines each), differing only in `event_type` and `payload`.

This duplication normalizes a pattern where service-layer code manages its own database engine lifecycle outside the canonical pool. If a third email-like event appears (e.g., `send_welcome_email` on registration), the pattern will be copy-pasted a third time.

## Goals / Non-Goals

**Goals:**
- Extract the engine-lifecycle + outbox-call pattern into a single private method on `AuthService`.
- Eliminate the duplicate `create_async_engine` / `dispose` blocks from `resend_verification` and `forgot_password`.
- Keep the change contained to `AuthService` (no architectural spill).

**Non-Goals:**
- No change to how the outbox module works.
- No injection of a PG session factory into `AuthService.__init__` (that would push PG awareness into every auth route for the sake of two edge cases).
- No change to any other service or module.

## Decisions

### Decision 1: Private instance method, not a module-level helper

**Choice:** Add `_publish_outbox_event(self, aggregate_type: str, aggregate_id: str, event_type: str, payload: dict[str, object]) -> None` as a private instance method on `AuthService`.

**Rationale:** The method needs no instance state beyond what's already available. A private method is the minimum viable extraction — it documents the pattern, centralizes the lifecycle, and is trivially promotable to a shared utility if a third caller appears. A module-level helper would require importing `AuthService` internals, which is backwards.

### Decision 2: engine lifecycle inside the helper, not injected

**Choice:** The helper manages `create_async_engine` + `dispose` internally, exactly as the duplicated blocks do now.

**Rationale:** `AuthService` does not have a PG session factory, and injecting one into the constructor would add a PG runtime dependency to every auth route. The helper's engine is created and disposed per-call, which is acceptable for low-frequency operations (user-initiated email resets, not hot-path). The `async with` pattern guarantees disposal even on exception.

**Alternatives considered:**
- *Inject `session_factory` into `AuthService.__init__`.* This would require the auth router or dependency to resolve a PG session factory, coupling auth to PostgreSQL even though auth's primary store is MongoDB. Rejected for architectural discipline.
- *Use the app-state session factory from FastAPI lifespan.* `AuthService` is not a FastAPI dependency — it's composed in the auth router via `Depends()`. Passing `app.state` through would create a coupling to FastAPI internals. Rejected.
- *Shared utility function at `app.shared.outbox` level.* Theoretically cleanest, but premature for two callers. Add when a third appears.

## Risks / Trade-offs

- **[Low] Per-call engine create/dispose overhead.** The engine creation and disposal has measurable cost (connection handshake, auth). For `resend_verification` and `forgot_password` this is acceptable — these are user-initiated operations at human timescales (seconds-to-minutes). If a high-frequency outbox caller appears in the auth service, promote to a shared utility with an injected engine.
- **[None] No behavioral change.** The helper replicates the exact same engine lifecycle as the duplicated blocks. Rollback is trivial: inline the helper back.
