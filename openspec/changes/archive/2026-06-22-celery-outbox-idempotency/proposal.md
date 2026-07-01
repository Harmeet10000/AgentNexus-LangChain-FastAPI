## Why

The current architecture has a reliability gap: the service layer writes to the DB, then calls `celery_app.send_task()` in a separate step. If `send_task()` fails (RabbitMQ down, network partition), the DB commit succeeds but the Celery task is never published. The document is stuck in "pending" state with no way to recover. The outbox pattern fixes this by making the DB write and event publish atomic.

## What Changes

### Transactional Outbox
- New `outbox_events` table: `id`, `aggregate_type`, `aggregate_id`, `event_type`, `payload`, `created_at`, `published_at`
- `with_outbox(tx, aggregate, event)` helper that writes the outbox row inside the same DB transaction as the business write, then calls `pg_notify('outbox_channel', event_id)` in the same transaction
- Relay process: uses PostgreSQL NOTIFY/LISTEN via `asyncpg-listen` library — subscribes to `outbox_channel`, publishes to Celery on notification
- Startup scan: relay catches any unpublished events created while offline
- Dead-letter after 5 failed publish attempts

### Affected Call Sites
- `search/service.py` `ingest_document()`: wrap `celery_app.send_task()` with outbox write
- `documents/service.py` `upload_document()`: wrap `celery_app.send_task()` with outbox write
- `auth/service.py` `resend_verification()`, `forgot_password()`: wrap email task sends

## Capabilities

### New Capabilities
- `transactional-outbox`: Atomic DB write + event publish via outbox table + relay

### Modified Capabilities
- (none)

## Impact

### Affected Code
- `src/app/shared/outbox/` — new module (model, relay, helper)
- `src/app/features/search/service.py` — replace `send_task` with outbox write
- `src/app/features/documents/service.py` — replace `send_task` with outbox write
- `src/app/features/auth/service.py` — replace `send_task` with outbox write
- `src/alembic/migrations/` — new migration for `outbox_events` table
- `src/app/lifecycle/lifespan.py` — start/stop relay on app lifecycle

### Affected APIs
- No breaking changes to request/response contracts
- Ingestion endpoints now return immediately (task is async via outbox)

### Dependencies Added
- `asyncpg-listen` — asyncpg LISTEN connection management with auto-reconnect

### Systems
- CI: new migration must pass `alembic upgrade head`
