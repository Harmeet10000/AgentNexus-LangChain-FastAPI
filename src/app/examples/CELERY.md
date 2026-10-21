# Celery documentation index

The canonical Celery documentation lives in the docs site:

- [Celery runtime architecture](/architecture/celery-runtime) — producer, JobBus, outbox relay, RabbitMQ, role isolation, process lifetimes, and at-least-once execution.
- [JobBus](/architecture/job-bus) — immutable catalog, direct and transactional seams, receipts, ownership, and compatible in-memory adapters.
- [Celery configuration reference](/configuration/celery) — queues, exchanges, DLX, roles, process shape, delivery settings, retries, Beat, commands, and completeness.
- [Add a Celery job](/guides/add-celery-job) — wire name, payload, catalog definition, canonical `app.worker.jobs` task adapter, routing, schedules, and checks.
- [Outbox pattern](/architecture/outbox-pattern) — transactional staging and relay behavior.
- [Environment variables](/configuration/environment-variables) — settings defaults and deployment overrides.

## Current source map

| Concern | Source of truth |
| --- | --- |
| Stable names, payloads, immutable definitions | `src/app/shared/jobs/names.py`, `payloads.py`, `catalog.py` |
| Producer contract and receipts | `src/app/shared/jobs/bus.py` |
| In-memory contract tests | `src/app/shared/jobs/in_memory.py` |
| Exchanges and catalog-derived routes | `src/app/shared/jobs/topology.py` |
| Producer-only Celery adapter | `src/app/connections/celery_producer.py` |
| Transactional outbox adapter and relay | `src/app/events/writer.py`, `src/app/events/relay.py` |
| Worker composition, role profiles, CLI | `src/app/worker/celery/app.py`, `profiles.py`, `cli.py` |
| Task base, retries, idempotency | `src/app/worker/celery/task.py` |
| Canonical task declarations | `src/app/worker/jobs/` |
| Derived module map and completeness | `src/app/worker/celery/registration.py`, `completeness.py` |
| Beat entries | `src/app/worker/celery/schedules.py` |
| Environment defaults | `src/app/config/settings.py` |

## Operational warning

Delivery is at least once. A broker receipt is not task completion, and a redelivered message can execute a side effect again. Use the Redis-backed idempotency helpers on `ResilientTask` and durable domain state. The `rpc://` result backend is ephemeral, and Beat must run as one operational singleton; neither is a durable workflow ledger or enforced leader election.

Supported local commands are `make celery`, `make celery-ingestion`, `make celery-crawler`, `make celery-beat`, and `make celery-completeness`. Worker roles select their own queue, prefork pool, concurrency, and child recycling; do not revive removed compatibility startup paths.
