## 1. Outbox Table & Migration

- [ ] 1.1 Create `src/app/shared/outbox/__init__.py`
- [ ] 1.2 Create `src/app/shared/outbox/model.py` with `OutboxEvent` SQLAlchemy model
- [ ] 1.3 Create `src/app/shared/outbox/dead_letter.py` with `DeadLetterEvent` model
- [ ] 1.4 Create Alembic migration: `outbox_events` table with index
- [ ] 1.5 Create Alembic migration: `dead_letter_events` table
- [ ] 1.6 Verify migration: `uv run alembic upgrade head && uv run alembic downgrade -1 && uv run alembic upgrade head`

## 2. Outbox Helper

- [ ] 2.1 Create `src/app/shared/outbox/helper.py` with `with_outbox()` function
- [ ] 2.2 `with_outbox()` accepts `AsyncSession`, `aggregate_type`, `aggregate_id`, `event_type`, `payload`
- [ ] 2.3 `with_outbox()` writes `OutboxEvent` row in the provided session (no commit — caller controls transaction)
- [ ] 2.4 Add unit test: `with_outbox()` creates outbox row with correct fields
- [ ] 2.5 Add unit test: outbox row is rolled back if business write fails

## 3. Relay Process

- [ ] 3.1 Create `src/app/shared/outbox/relay.py` with `OutboxRelay` class
- [ ] 3.2 `OutboxRelay.poll()` method: SELECT unpublished events with `FOR UPDATE SKIP LOCKED`
- [ ] 3.3 `OutboxRelay.publish()` method: call `celery_app.send_task()` with payload
- [ ] 3.4 `OutboxRelay.run()` method: poll loop with 250ms interval
- [ ] 3.5 On publish success: set `published_at = now()`
- [ ] 3.6 On publish failure: increment `publish_attempts`, set `last_error`
- [ ] 3.7 After 5 failures: move to `dead_letter_events` table, log ERROR
- [ ] 3.8 Add processing timeout: events stuck >30s in "processing" state are re-queued

## 4. Relay Lifecycle

- [ ] 4.1 Start relay in `lifespan.py` as `asyncio.Task` after all deps ready
- [ ] 4.2 Store relay task in `app.state.outbox_relay`
- [ ] 4.3 On shutdown: cancel relay task, wait 5s for drain
- [ ] 4.4 Log relay start/stop with event count

## 5. Call Site Migration

- [ ] 5.1 `search/service.py:ingest_document()`: replace `celery_app.send_task()` with `with_outbox()`
- [ ] 5.2 `documents/service.py:upload_document()`: replace `celery_app.send_task()` with `with_outbox()`
- [ ] 5.3 `auth/service.py:resend_verification()`: replace `send_verification_email.delay()` with `with_outbox()`
- [ ] 5.4 `auth/service.py:forgot_password()`: replace `send_password_reset_email.delay()` with `with_outbox()`
- [ ] 5.5 Remove old `celery_app.send_task()` calls (keep task definitions)

## 6. Dead Letter Replay

- [ ] 6.1 Create `scripts/replay_outbox.py` CLI: reads dead-lettered events, re-publishes to Celery
- [ ] 6.2 Add `uv run outbox:replay` script entry in `pyproject.toml`

## 7. Testing

- [ ] 7.1 Add integration test: outbox row created in same transaction as business write
- [ ] 7.2 Add integration test: relay publishes pending event within 250ms
- [ ] 7.3 Add integration test: failed publish retries up to 5 times
- [ ] 7.4 Add integration test: dead-lettered event is logged and replayable
- [ ] 7.5 Run `uv run pytest tests/ -v`

## 8. Lint & Type Check

- [ ] 8.1 Run `uv run ruff check src/app/shared/outbox/`
- [ ] 8.2 Run `uv run ty check src/app/shared/outbox/`
