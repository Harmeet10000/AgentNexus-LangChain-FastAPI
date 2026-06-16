## Context

The project has 9 Celery tasks across 5 files. The two highest-volume tasks are:
- `tasks.search_ingest` — called from `search/service.py:89` via `celery_app.send_task()`
- `tasks.documents_ingest` — called from `documents/service.py` via `celery_app.send_task()`

Both are called AFTER the DB commit succeeds. If `send_task()` fails, the DB row exists but the task is never queued. The Celery tasks already have idempotency via Redis locks (`ResilientTask` base class), but that only prevents duplicate processing — it doesn't solve the "task never published" problem.

The outbox pattern solves this by writing the event to a DB row in the SAME transaction as the business write, then a relay process publishes it to RabbitMQ.

## Goals / Non-Goals

**Goals:**
- Atomic DB write + event publish via outbox table
- Relay process that polls and publishes to Celery
- Dead-letter after 5 failed publish attempts
- Zero breaking changes to existing API contracts

**Non-Goals:**
- Replace Celery with another task queue
- Change the existing idempotency locks (they work fine)
- Add event sourcing (outbox is for reliable delivery, not audit trail)
- Add Kafka/RabbitMQ transactional outbox (keep it DB-based for simplicity)

## Decisions

### D1: Outbox table design — simple polling, not CDC

**Decision:** Single `outbox_events` table with `published_at` column. Relay polls every 250ms with `SELECT ... WHERE published_at IS NULL ORDER BY created_at LIMIT 10 FOR UPDATE SKIP LOCKED`.

**Rationale:** Polling is simple, battle-tested, and sufficient for the volume (~100 tasks/day). CDC (Debezium) adds operational complexity. `FOR UPDATE SKIP LOCKED` prevents duplicate publishing across relay instances.

**Alternatives considered:**
- *CDC (Debezium)*: overkill for this volume — rejected
- *RabbitMQ transactions*: coupling to broker — rejected
- *In-memory outbox*: lost on restart — rejected

### D2: Relay lifecycle — started in lifespan, runs in background task

**Decision:** Relay runs as an `asyncio.Task` started in `lifespan.py` shutdown hook. It polls every 250ms, publishes pending events, marks as published. On shutdown, it drains remaining events (best-effort).

**Rationale:** Running the relay in the FastAPI process avoids deploying a separate service. The relay is lightweight (one SELECT + one publish per tick). On graceful shutdown, it has a 5s window to drain.

**Alternatives considered:**
- *Separate relay service*: adds deployment complexity — rejected for now
- *Celery beat schedule*: too coarse (1min minimum) — rejected
- *Sync in request path*: blocks the response — rejected

### D3: Dead-letter after 5 failures

**Decision:** Track `publish_attempts` in the outbox row. After 5 failed publishes, move to a `dead_letter_events` table. Alert via logging.

**Rationale:** Infinite retries waste resources. 5 attempts covers transient failures (RabbitMQ restart ~30s). Dead-letter table allows manual inspection and replay.

### D4: Idempotency — outbox key includes aggregate

**Decision:** Outbox event includes `aggregate_type` + `aggregate_id` + `event_type`. The relay checks for duplicate `event_type` on the same aggregate before publishing.

**Rationale:** Prevents duplicate events if the same document is ingested twice (idempotency lock already handles this at the Celery level, but the outbox should also be safe).

## Risks / Trade-offs

- **[Polling overhead]** 250ms polling = ~4 queries/second. **Mitigation:** `FOR UPDATE SKIP LOCKED` is lightweight; idle polls return 0 rows in <1ms.
- [**Relay crash**] If the relay crashes mid-publish, the event stays in "processing" state. **Mitigation:** Add a `processing_timeout_seconds` (30s) — events stuck in processing are re-queued.
- **[DB connection pool]** Relay uses one async session from the pool. **Mitigation:** Relay session is short-lived (SELECT + UPDATE + COMMIT in <10ms).
