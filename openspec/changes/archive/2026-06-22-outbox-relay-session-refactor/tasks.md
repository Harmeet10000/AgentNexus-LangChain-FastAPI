## 1. Add `_MAX_RETRIES` constant

- [x] 1.1 Add `_MAX_RETRIES: Final[int] = 5` as a module-level constant in `src/app/shared/outbox/relay.py` (import `Final` from `typing`)
- [x] 1.2 Replace `publish_attempts < 5` in `run_startup_scan` SQL with `publish_attempts < :max_retries` and bind `{"max_retries": _MAX_RETRIES}`
- [x] 1.3 Replace `publish_attempts < 5` in `_handle_notification` SQL with `publish_attempts < :max_retries` and bind `{"max_retries": _MAX_RETRIES}`
- [x] 1.4 Replace `if attempts >= 5` in `_mark_failed` with `if attempts >= _MAX_RETRIES`

## 2. Remove dead `_running` flag and `shutdown()` method

- [x] 2.1 Remove `self._running = False` from `__init__`
- [x] 2.2 Remove `self._running` field declaration
- [x] 2.3 Remove `shutdown()` method entirely
- [x] 2.4 Remove `self._running = True` from `run_listener`
- [x] 2.5 Update `lifespan.py` shutdown block: remove `await app.state.outbox_relay.shutdown()` call, keep only `app.state.outbox_relay_task.cancel()`

## 3. Make `session` required in `_publish`, `_mark_published`, `_mark_failed`

- [x] 3.1 Change `run_startup_scan` to use `self._session_factory()` instead of `create_async_engine()`. Remove the `engine` create/dispose — use `async with self._session_factory() as session:` instead of `async with engine.begin() as conn:`. Pass `session` to each `await self._publish(dict(row), session=session)` call.
- [x] 3.2 Change `_publish` signature: `session: AsyncSession | None = None` → `session: AsyncSession`. Remove the optionality. Update docstring.
- [x] 3.3 Change `_mark_published` signature: `session: AsyncSession | None = None` → `session: AsyncSession`. Remove the `if session is not None: ... else: ...` block — keep only the session branch. Remove the `else` branch entirely.
- [x] 3.4 Change `_mark_failed` signature: `session: AsyncSession | None = None` → `session: AsyncSession`. Remove the `if session is not None: ... else: ...` block — keep only the session branch. Remove the `else: return await self._mark_failed(...)` recursive fallback entirely.
- [x] 3.5 Verify `_handle_notification` already passes `session` to `_publish` (line 107: `await self._publish(dict(result), session=session)`) — no change needed.

## 4. Run checks and verify

- [x] 4.1 Run `uv run ruff check src/app/shared/outbox/relay.py` — ensure no lint regressions
- [x] 4.2 Run `uv run ty check src/app/shared/outbox/relay.py` — ensure no type regressions
- [x] 4.3 Run `uv run python tests/unit/test_outbox.py` — ensure all tests pass (pytest conftest has pre-existing circular import, bypassed with direct python execution)
- [x] 4.4 Run `uv run ruff format src/app/shared/outbox/relay.py` — ensure formatting is clean
