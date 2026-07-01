# Capability: transactional-outbox

## Purpose

Ensure atomic DB write + Celery task publish via a transactional outbox pattern. Prevents "task never published" failures when RabbitMQ is temporarily unavailable.

## ADDED Requirements

### Requirement: Outbox Table Schema
#### Scenario: Table exists
- **WHEN** migration runs
- **THEN** `outbox_events` and `dead_letter_events` tables exist

### Requirement: Outbox Helper
#### Scenario: Helper inserts row
- **WHEN** `with_outbox()` is called
- **THEN** an outbox row is inserted and `pg_notify` fires

### Requirement: Relay Process
#### Scenario: Notification processed
- **WHEN** a notification arrives
- **THEN** the relay publishes the event or dead-letters after 5 failures

### Requirement: Relay Lifecycle
#### Scenario: Lifecycle managed
- **WHEN** the app starts
- **THEN** the relay starts after deps are ready and drains on shutdown

### Requirement: Dead Letter
#### Scenario: Dead letter
- **WHEN** publish fails 5 times
- **THEN** the event is moved to `dead_letter_events`

### Requirement: Migration
#### Scenario: Migration runs
- **WHEN** alembic upgrade is run
- **THEN** both tables are created idempotently

## Requirements

### R1: Outbox Table Schema
```sql
CREATE TABLE outbox_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type VARCHAR(64) NOT NULL,    -- 'search_document', 'user_document', 'auth_email'
    aggregate_id VARCHAR(128) NOT NULL,     -- document_id, user_id
    event_type VARCHAR(64) NOT NULL,        -- 'search_ingest', 'documents_ingest', 'send_verification_email'
    payload JSONB NOT NULL,                 -- task kwargs
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ,
    publish_attempts INT NOT NULL DEFAULT 0,
    last_error TEXT
);

CREATE INDEX idx_outbox_unpublished ON outbox_events (created_at)
    WHERE published_at IS NULL;
```

### R2: Outbox Helper
- `with_outbox(session, aggregate_type, aggregate_id, event_type, payload)` — writes outbox row inside existing transaction
- Uses SQLAlchemy `AsyncSession` (same transaction as business write)
- After INSERT and flush, call `SELECT pg_notify('outbox_channel', :event_id)` inside the **same transaction** — `pg_notify` is transactional, so the notification is only delivered after the business write commits
- Called from service layer instead of `celery_app.send_task()`
- Returns the event ID (UUID string)

### R3: Relay Process — NOTIFY/LISTEN (not polling)
- Uses `asyncpg-listen` library for PostgreSQL LISTEN connection management (auto-reconnect)
- Subscribe to `outbox_channel` at relay startup
- On notification: parse event_id from payload, SELECT the row by ID:
  ```sql
  SELECT * FROM outbox_events
  WHERE id = :event_id
    AND published_at IS NULL
    AND publish_attempts < 5
  FOR UPDATE SKIP LOCKED
  ```
- If the SELECT returns a row: call `celery_app.send_task()` with payload
- On success: `UPDATE outbox_events SET published_at = now() WHERE id = :event_id`
- On failure: `UPDATE outbox_events SET publish_attempts = publish_attempts + 1, last_error = :error WHERE id = :event_id`
- After 5 failures: move row to `dead_letter_events` table

#### Startup scan
- On relay start, run a one-time scan for unpublished events:
  ```sql
  SELECT * FROM outbox_events
  WHERE published_at IS NULL
    AND publish_attempts < 5
  ORDER BY created_at
  LIMIT 100
  FOR UPDATE SKIP LOCKED
  ```
- Process all found events (same logic as notification path)
- After the scan, switch to pure LISTEN mode — no periodic polling
- This catches events created while the relay was offline

### R4: Relay Lifecycle
- Start in `lifespan.py` after all deps are ready
- Establish asyncpg-listen connection using the same DATABASE_URL as the app
- Run startup scan, then enter listen loop
- On shutdown: stop the listen task with 5s drain window
- Log relay start/stop, daily event counts

### R5: Dead Letter
- After 5 failed publishes, move to `dead_letter_events` table
- Log ERROR with full payload for manual inspection
- Provide `bun run outbox:replay <event_id>` CLI to replay dead-lettered events

### R6: Migration
- Alembic migration for `outbox_events` + `dead_letter_events` tables
- Idempotent migration (safe to run multiple times)

## Dependencies Added
- `asyncpg-listen` — PyPI library for asyncpg LISTEN connection management with auto-reconnect

## Acceptance Criteria
- [ ] `with_outbox()` writes outbox row + sends `pg_notify` in same transaction
- [ ] Relay receives notification via asyncpg-listen and publishes within 100ms
- [ ] Startup scan picks up events created while relay was offline
- [ ] Failed publishes retry up to 5 times, then dead-letter
- [ ] Dead-lettered events are logged and replayable
- [ ] Relay drains on graceful shutdown
- [ ] No breaking changes to existing API contracts

## Non-Goals
- CDC-based outbox (Debezium)
- Separate relay service deployment
- Event sourcing
- Kafka/RabbitMQ transactional outbox
- Periodic polling fallback (pure LISTEN + startup scan only)
