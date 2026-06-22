## 1. Extract `_publish_outbox_event` helper

- [x] 1.1 Add private method `_publish_outbox_event(self, aggregate_type: str, aggregate_id: str, event_type: str, payload: dict[str, object]) -> None` to `AuthService`
- [x] 1.2 Inside `_publish_outbox_event`: import `create_async_engine`, `AsyncSession`, `get_database_url`, and `with_outbox` — then create engine, wrap in session, call `with_outbox`, dispose in `finally`

## 2. Migrate callers

- [x] 2.1 In `resend_verification`, replace the 8-line engine/create/dispose block with a single call: `self._publish_outbox_event(aggregate_type="auth_email", aggregate_id=str(resolved.id), event_type="auth.send_verification_email", payload={"user_id": str(resolved.id), "email": resolved.email, "token": new_token})`. Remove the deleted imports if no longer needed elsewhere.
- [x] 2.2 In `forgot_password`, replace the 8-line engine/create/dispose block with a single call: `self._publish_outbox_event(aggregate_type="auth_email", aggregate_id=str(resolved.id), event_type="auth.send_password_reset_email", payload={"user_id": str(resolved.id), "email": resolved.email, "token": reset_token})`. Remove the deleted imports if no longer needed elsewhere.

## 3. Remove stale imports

- [x] 3.1 After migration, check if `create_async_engine`, `AsyncSession`, `get_database_url`, and `with_outbox` are still explicitly imported at the top of `resend_verification`/`forgot_password` — remove any leftover inline imports from those method bodies
- [x] 3.2 Run `ruff check --fix src/app/features/auth/service.py` to clean up any unused imports

## 4. Run checks and verify

- [x] 4.1 Run `uv run ruff check src/app/features/auth/service.py` — ensure no lint regressions (only pre-existing S105/S106 false positives remain)
- [x] 4.2 Run `uv run ty check src/app/features/auth/service.py` — ensure no type regressions (only pre-existing warnings)
- [x] 4.3 Run `uv run ruff format src/app/features/auth/service.py` — ensure formatting is clean
