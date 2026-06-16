# Capability: transactional-outbox

## Purpose

Ensure atomic DB write + Celery task publish via a transactional outbox pattern. Prevents "task never published" failures when RabbitMQ is temporarily unavailable.

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
- `with_outbox(tx, aggregate_type, aggregate_id, event_type, payload)` — writes outbox row inside existing transaction
- Uses SQLAlchemy `AsyncSession` (same transaction as business write)
- Called from service layer instead of `celery_app.send_task()`

### R3: Relay Process
- Runs as `asyncio.Task` in FastAPI process
- Polls every 250ms: `SELECT ... WHERE published_at IS NULL AND publish_attempts < 5 ORDER BY created_at LIMIT 10 FOR UPDATE SKIP LOCKED`
- For each row: call `celery_app.send_task()` with payload from `outbox_events.payload`
- On success: set `published_at = now()`
- On failure: increment `publish_attempts`, set `last_error`
- After 5 failures: move to `dead_letter_events` table (same schema + `dead_letter_at`)

### R4: Relay Lifecycle
- Start in `lifespan.py` after all deps are ready
- Stop on shutdown with 5s drain window
- Log per-publish success/failure with timing

### R5: Dead Letter
- After 5 failed publishes, move to `dead_letter_events` table
- Log ERROR with full payload for manual inspection
- Provide `bun run outbox:replay <event_id>` CLI to replay dead-lettered events

### R6: Migration
- Alembic migration for `outbox_events` + `dead_letter_events` tables
- Idempotent migration (safe to run multiple times)

## Acceptance Criteria
- [ ] `with_outbox()` writes outbox row in same transaction as business data
- [ ] Relay publishes pending events within 250ms
- [ ] Failed publishes retry up to 5 times
- [ ] Dead-lettered events are logged and replayable
- [ ] Relay drains on graceful shutdown
- [ ] No breaking changes to existing API contracts

## Non-Goals
- CDC-based outbox (Debezium)
- Separate relay service deployment
- Event sourcing
- Kafka/RabbitMQ transactional outbox
