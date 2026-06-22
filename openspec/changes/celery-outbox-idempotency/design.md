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

### D1: Outbox table design — NOTIFY/LISTEN, not polling

**Decision:** Single `outbox_events` table. Relay uses PostgreSQL NOTIFY/LISTEN via the `asyncpg-listen` library instead of polling. After writing the outbox row, `with_outbox()` calls `pg_notify('outbox_channel', event_id)` inside the same transaction.

**Rationale:** pg_notify() is transactional — the notification is queued and only delivered when the containing transaction commits. This gives zero-latency wakeup with zero idle query overhead. No periodic polling needed.

**Alternatives considered:**
- *250ms polling*: ~4 idle queries/sec at current volume (~100 tasks/day) — negligible overhead but inelegant — rejected
- *CDC (Debezium)*: overkill for this volume — rejected
- *In-memory outbox*: lost on restart — rejected

### D2: Relay lifecycle — started in lifespan, runs in background task

**Decision:** Relay runs as an `asyncio.Task` started in `lifespan.py`. On start: run a one-time scan for unpublished events (catch events created while offline), then subscribe to `outbox_channel` via asyncpg-listen. On shutdown: drain remaining events (best-effort, 5s window).

**Rationale:** Running the relay in the FastAPI process avoids deploying a separate service. The asyncpg-listen connection is idle until a notification arrives. On graceful shutdown, it has a 5s window to finish in-flight publishes.

**Alternatives considered:**
- *Separate relay service*: adds deployment complexity — rejected for now
- *Celery beat schedule*: too coarse for notification-driven — rejected
- *250ms polling timer*: extra queries against database — rejected in favor of LISTEN

### D3: Dead-letter after 5 failures

**Decision:** Track `publish_attempts` in the outbox row. After 5 failed publishes, move to a `dead_letter_events` table. Alert via logging.

**Rationale:** Infinite retries waste resources. 5 attempts covers transient failures (RabbitMQ restart ~30s). Dead-letter table allows manual inspection and replay.

### D4: Idempotency — outbox key includes aggregate

**Decision:** Outbox event includes `aggregate_type` + `aggregate_id` + `event_type`. The relay checks for duplicate `event_type` on the same aggregate before publishing.

**Rationale:** Prevents duplicate events if the same document is ingested twice (idempotency lock already handles this at the Celery level, but the outbox should also be safe).

### D5: NOTIFY timing — inside the INSERT transaction

**Decision:** `with_outbox()` calls `SELECT pg_notify('outbox_channel', :event_id)` inside the same transaction as the outbox INSERT. `pg_notify` is transactional — the notification is queued in memory and only delivered to listening sessions after the transaction commits.

**Rationale:** Zero crash window. If the transaction rolls back (e.g. business write fails), the notification is never sent — no orphan NOTIFY. If the transaction commits, the row is visible and the notification arrives. No edge case where the row exists but nobody knows about it.

**Alternatives considered:**
- *Two-step (INSERT then NOTIFY after commit)*: crash window between commit and NOTIFY — rejected
- *NOTIFY-only (no outbox table)*: no persistence, fail on crash — rejected

### D6: Connection management via asyncpg-listen

**Decision:** Use the `asyncpg-listen` PyPI library to manage the LISTEN connection. It connects to the same DATABASE_URL as the app, subscribes to `outbox_channel`, and delivers notifications via an async generator. Automatic reconnection on connection drops.

**Rationale:** asyncpg-listen handles the edge cases (reconnect after DB restart, cleanup of stale connections) that a raw connection doesn't. A raw asyncpg connection outside the pool would need manual reconnection logic.

**Alternatives considered:**
- *Raw asyncpg connection*: no reconnection handling — rejected
- *SQLAlchemy pool connection*: can't LISTEN on a pooled connection (channel state leaks) — rejected

## Risks / Trade-offs

- **[Pure LISTEN — no fallback poll]** If the relay misses a notification (connection drop, buffer overflow), the event is stuck until relay restart. **Mitigation:** The startup scan catches all unpublished events. Between restarts, the window is bounded. Acceptable at current volume (~100 tasks/day).
- **[Relay crash mid-publish]** The event stays in "processing" state. **Mitigation:** SELECT uses `FOR UPDATE SKIP LOCKED` so a crashed transaction releases the lock. Next relay start picks it up via startup scan.
- **[asyncpg-listen dependency]** Adds one external library. **Mitigation:** Lightweight, well-maintained, wraps standard asyncpg connect/listen/notify API.
- **[DB connection usage]:** asyncpg-listen maintains its own connection (not from the pool). **Mitigation:** One idle connection per relay instance. Negligible.
