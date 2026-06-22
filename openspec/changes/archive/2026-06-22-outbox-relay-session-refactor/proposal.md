## Why

`OutboxRelay._publish()` accepts an optional `AsyncSession | None` parameter, forcing every downstream method (`_mark_published`, `_mark_failed`) to duplicate their entire logic path in an `if session is not None: ... else: ...` branch. `_mark_failed` compounds the problem with a recursive fallback call when no session is provided. Separately, the retry limit `5` is hardcoded in three SQL sites (two WHERE clauses in `run_startup_scan` and `_handle_notification`, plus a `>= 5` comparison in `_mark_failed`) with zero discoverability — changing it requires editing three separate locations that are easy to miss. The relay's `shutdown()` method only sets a `_running` flag, but `run_listener` uses `asyncpg_listen` with `NO_TIMEOUT` and never checks this flag, making shutdown purely cosmetic — only `asyncio.Task.cancel()` actually stops it. These three issues together make the relay harder to maintain, unsafe to tune, and misleading to call.

## What Changes

- **Make session required in `_publish`, `_mark_published`, `_mark_failed`.** Push session creation up to the two callers (`run_startup_scan` already has an engine; `_handle_notification` already has a session factory). This eliminates the `AsyncSession | None` optionality, deletes the `if session is not None` branching in both methods, and removes the recursive fallback in `_mark_failed`.
- **Extract `_MAX_RETRIES: Final[int] = 5` as a module-level constant.** Replace all three hardcoded `5` references in SQL and Python. Use the constant name in SQL inline comments so the bound is visible when reading queries.
- **Make `shutdown()` actually stop the listener, or delete the dead `_running` machinery.** Either switch `run_listener` to a periodic notification timeout (e.g., 1.0s) with a `_running` check in the handler, or remove `self._running`, `shutdown()`, and the field entirely since `asyncio.Task.cancel()` is the real shutdown mechanism.
- **No spec-level capability changes** — all modifications are internal refactoring of `relay.py` only.

## Capabilities

### New Capabilities
*(none — internal refactoring only)*

### Modified Capabilities
*(none — no spec-level behavior changes)*

## Impact

- **File:** `src/app/shared/outbox/relay.py` — restructured `_publish`, `_mark_published`, `_mark_failed` to require session; added `_MAX_RETRIES` constant; either cleaned up or wired `shutdown()` logic.
- **Test file:** `tests/unit/test_outbox.py` — update mocks to pass a session to `_publish`-equivalent paths if any are tested.
- **Lifespan file:** `src/app/lifecycle/lifespan.py` — update outbox relay shutdown if `shutdown()` changes signature or is removed.
- **No new dependencies.** No database migrations. No external API changes.
