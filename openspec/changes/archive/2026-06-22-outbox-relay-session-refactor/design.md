## Context

`OutboxRelay` in `src/app/shared/outbox/relay.py` (190 lines) was implemented with an optional `session` parameter threaded through `_publish` → `_mark_published` / `_mark_failed`. The intent was twofold: (a) when called from `_handle_notification` (the listen loop already has a session from the factory), pass it through for transactional consistency; (b) when called from `run_startup_scan` (which only has a raw engine), let the downstream methods create their own session. This design forces every downstream method to branch on `session is not None`, duplicating ~30 lines across two methods.

Additionally, the retry threshold `5` is embedded as a raw integer in three SQL/Python locations with no named constant, and the `_running` flag / `shutdown()` pattern is purely cosmetic — the `asyncpg_listen.NotificationListener` with `NO_TIMEOUT` never checks the flag.

## Goals / Non-Goals

**Goals:**
- Eliminate the `if session is not None: ... else: ...` duplication in `_mark_published` and `_mark_failed`.
- Remove the recursive fallback call in `_mark_failed`.
- Make session lifetime unambiguous and enforced by the type system (required parameter, not optional).
- Extract a single `_MAX_RETRIES: Final[int] = 5` constant and reference it everywhere.
- Either make `shutdown()` actually stop the listener, or remove the dead `_running` machinery.

**Non-Goals:**
- No change to the public API of `OutboxRelay` (`run_startup_scan`, `run_listener`, `shutdown`).
- No change to the outbox model, `with_outbox`, or the database schema.
- No change to the lifespan wiring.
- No change to retry semantics (the value stays 5, just extracted to a named constant).

## Decisions

### Decision 1: Require session in `_publish`, push session creation to callers

**Choice:** Make `session: AsyncSession` a required (non-optional) parameter on `_publish`, `_mark_published`, and `_mark_failed`. The two callers adjust:
- `_handle_notification` already creates `async with self._session_factory() as session:` — it passes `session` through unchanged.
- `run_startup_scan` already creates `async with engine.begin() as conn:` — change this to `async with self._session_factory() as session:` (the relay already has `self._session_factory` from `__init__`, so no new infrastructure needed). This removes the need to create a separate engine in `run_startup_scan`.

**Rationale:** The core insight is that `run_startup_scan` already receives `self._session_factory` via the constructor (line 30). It does not need to create its own engine — it can use `self._session_factory()` directly. This eliminates the only reason session was optional in the first place.

**Alternatives considered:**
- *Keep optional session, factor the shared SQL into a private helper that accepts a session.* This does not fix the root cause — the branching stays, it just moves to a smaller wrapper.
- *Use a context manager or decorator to manage session creation.* Adds indirection for no benefit; the two callers have different session needs.

### Decision 2: Replace `engine.dispose()` in `run_startup_scan` with `self._session_factory`

**Choice:** Change `run_startup_scan` from `create_async_engine()` + `engine.begin()` + `engine.dispose()` to `self._session_factory()` + `session.begin()` + implicit cleanup via `async with`.

**Rationale:** `self._session_factory` uses the same `async_sessionmaker` that the rest of the app uses. Creating a second engine per startup scan is wasteful and was only done because the session-factory approach was not tried first.

### Decision 3: Module-level `_MAX_RETRIES` constant

**Choice:** Add `_MAX_RETRIES: Final[int] = 5` at the top of `relay.py`. Replace all three `5` references. Add PostgreSQL inline comments: `publish_attempts < :max_retries  /* MAX_RETRIES */`.

**Rationale:** Named constants are discoverable and single-source-of-truth. The SQL bind parameter `:max_retries` makes the intent visible when reading raw queries.

### Decision 4: Either wire shutdown or delete dead code

**Option A (preferred):** Delete `self._running`, delete `shutdown()`. The shutdown sequence in `lifespan.py` already does `app.state.outbox_relay_task.cancel()` (line 268) which cancels the `asyncio.Task` wrapping `run_listener()`. The `_running` flag and `shutdown()` method add no value.

**Option B (fallback):** Keep `shutdown()`, wire it by setting `notification_timeout=1.0` and checking `self._running` in `_handle_notification`. This makes `shutdown()` effective within 1 second.

**Chosen:** Option A (delete dead code). Simpler, fewer moving parts, no periodic wake-up overhead.

## Risks / Trade-offs

- **[Low] `run_startup_scan` now uses session-factory sessions instead of `engine.begin()`.** Sessions from `async_sessionmaker` require explicit `await session.commit()` for writes. `run_startup_scan` only reads (SELECT), so no commit is needed — but any future write added here must remember to commit. Mitigation: current code is read-only; any future write change would be reviewed.
- **[Low] Deleting `shutdown()` removes the main-loop cancellation path.** `lifespan.py` already uses `asyncio.Task.cancel()`, so deletion is safe. Any external caller invoking `relay.shutdown()` would break — audit `lifespan.py` (the only caller) to confirm safe. Confirmed: only `lifespan.py` line 267 calls `relay.shutdown()`.
