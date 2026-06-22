## 0. Dependency

- [x] 0.1 Add `asyncpg-listen` to `pyproject.toml`
- [x] 0.2 Run `uv sync`

## 1. Outbox Table & Migration

- [x] 1.1 Create `src/app/shared/outbox/__init__.py`
- [x] 1.2 Create `src/app/shared/outbox/model.py` with `OutboxEvent` SQLAlchemy model
- [x] 1.3 Create `src/app/shared/outbox/dead_letter.py` with `DeadLetterEvent` model
- [x] 1.4 Create Alembic migration: `outbox_events` table + partial index on `(created_at) WHERE published_at IS NULL`
- [x] 1.5 Create Alembic migration: `dead_letter_events` table
- [~] 1.6 Verify migration: `uv run alembic upgrade head && uv run alembic downgrade -1 && uv run alembic upgrade head`  (requires running PG)

## 2. Outbox Helper

- [x] 2.1 Create `src/app/shared/outbox/helper.py` with `with_outbox()` function
- [x] 2.2 `with_outbox()` accepts `AsyncSession`, `aggregate_type`, `aggregate_id`, `event_type`, `payload`
- [x] 2.3 `with_outbox()` writes `OutboxEvent` row, then calls `SELECT pg_notify('outbox_channel', :event_id)` in the same session
- [x] 2.4 Build the `pg_notify` call as `text("SELECT pg_notify('outbox_channel', :event_id)")` with `{"event_id": str(event.id)}`
- [x] 2.5 Each `with_outbox()` call in the same transaction sends a separate notification
- [ ] 2.6 Add unit test: `with_outbox()` creates outbox row with correct fields
- [ ] 2.7 Add unit test: outbox row + notification are rolled back if business write fails

## 3. Relay Process

- [x] 3.1 Create `src/app/shared/outbox/relay.py` with `OutboxRelay` class
- [x] 3.2 `OutboxRelay.__init__()` accepts `database_url`, `celery_app`, `session_factory`
- [x] 3.3 **Startup scan**: on relay start, SELECT all unpublished events (LIMIT 100, `FOR UPDATE SKIP LOCKED`) and publish them
- [x] 3.4 **Listen loop**: use `asyncpg_listen.NotificationListener` to subscribe to `outbox_channel`. On each notification, parse the event_id from the payload
- [x] 3.5 **Fetch + publish**: SELECT the event by ID with `FOR UPDATE SKIP LOCKED`, call `celery_app.send_task()` with the stored payload
- [x] 3.6 On publish success: `UPDATE outbox_events SET published_at = now() WHERE id = :event_id`
- [x] 3.7 On publish failure: increment `publish_attempts`, set `last_error`
- [x] 3.8 After 5 failures: move row to `dead_letter_events` table, log ERROR

## 4. Relay Lifecycle

- [x] 4.1 Start relay in `lifespan.py` after all deps ready
- [x] 4.2 Run startup scan, then enter listen loop as `asyncio.Task`
- [x] 4.3 Store relay task in `app.state.outbox_relay`
- [x] 4.4 On shutdown: cancel relay task
- [x] 4.5 Log relay start/stop with event count

## 5. Call Site Migration

- [x] 5.1 `search/service.py:ingest_document()`: replace `celery_app.send_task()` with `with_outbox()`
- [x] 5.2 `documents/service.py:upload_document()`: replace `celery_app.send_task()` with `with_outbox()`
- [x] 5.3 `auth/service.py:resend_verification()`: replace `send_verification_email.delay()` with `with_outbox()`
- [x] 5.4 `auth/service.py:forgot_password()`: replace `send_password_reset_email.delay()` with `with_outbox()`
- [x] 5.5 Remove old `celery_app.send_task()` / `.delay()` calls (keep Celery task definitions)

## 6. Dead Letter Replay

- [ ] 6.1 Create `scripts/replay_outbox.py` CLI: reads dead-lettered events, re-publishes to Celery
- [ ] 6.2 Add `uv run outbox:replay` script entry in `pyproject.toml`

## 7. Testing

- [ ] 7.1 Add integration test: outbox row + notification created in same transaction as business write
- [ ] 7.2 Add integration test: relay receives notification via asyncpg-listen and publishes within 100ms
- [ ] 7.3 Add integration test: startup scan picks up events created while relay was offline
- [ ] 7.4 Add integration test: failed publish retries up to 5 times then dead-letters
- [ ] 7.5 Run `uv run pytest tests/ -v`

## 8. Lint & Type Check

- [x] 8.1 Run `uv run ruff check src/app/shared/outbox/`  (0 errors)
- [x] 8.2 Run `uv run ty check src/app/shared/outbox/`  (0 errors)
