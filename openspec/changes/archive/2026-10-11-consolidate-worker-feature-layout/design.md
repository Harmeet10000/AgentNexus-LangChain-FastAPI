> Change class: **L**.

## Context

See proposal.md for motivation. The repository already has nested settings, feature-owned models under billing, a central Base with constraint naming, explicit Alembic model registration, a lifecycle resource registry, and a transactional outbox. The supplied tree partly describes existing work and partly assumes obsolete paths. Current worker concerns are concentrated in a 1,300-line connection module and six top-level task modules.

## Goals / Non-Goals

**Goals:** A producer can validate/publish without importing consumers; workers own process resources; task modules sit beside use cases; the runtime is testable without real services.

**Non-Goals:** An all-features-independence fiction, renaming existing model.py/error modules for cosmetics, provider exactly-once guarantees, a broker switch, or database changes. Full extraction of Docling into a separately packaged deployment is a different dependency migration.

## Decisions

1. **Shared wire contracts, separate producer and worker applications.** `app.shared.jobs` holds names, payload models and typed validation. `app.connections.celery` is only the configured producer factory/application. `app.worker.celery_app` creates the worker application from that factory, attaches the task base, explicit includes, schedules and worker signals. A shared topology supplies identical queue/routing settings. Validation uses a static payload map; it never imports the declaring task module. A shared decorator that merely re-exports worker code would conceal, rather than fix, the dependency.
2. **Feature ownership follows the current domain.** Move auth, documents, crawler, credits, agent memory and billing tasks to existing features. Scheduled billing/credit orchestration moves to feature jobs modules; task adapters retain sync-to-async invocation. Keep billing's existing internal subfeatures. Do not force unrelated features through eager package exports.
3. **Persistent loop and process ownership.** Introduce a small worker runtime for generic async tasks and reliability operations. Preserve explicit capability resource lifetimes for document/crawler workers and move them under worker. Child signals reset inherited state without I/O. First task provisions slow resources; normal shutdown closes resources before loops. Prefork/solo are the supported execution models; thread/gevent/eventlet concurrency is outside this design.
4. **Preserve transport contracts.** Keep RabbitMQ quorum queues, publisher confirms, separate default/ingestion/crawler consumers, RPC results, time limits, retries and existing names. Fix the memory beat entry's positional argument mismatch with its keyword-only task signature. Existing outbox delivery remains authoritative. Redis leases are concurrency controls and never documented as proof of exactly-once external effects.
5. **Enforce actual boundaries.** Import-linter forbids API/producer/shared jobs paths from reaching worker or feature task modules, including indirect imports. Task modules are explicit allowed worker entry points, never imported by web code. Keep shared jobs free of feature internals. Add discovery-versus-registration and fresh-process producer tests. Do not add a global independence contract contradicted by billing orchestration and existing shared graph integrations.
6. **Packaging with evidence.** Correct first-party roots to real import packages (exclude the migration directory named alembic, which must not shadow installed Alembic). Keep pytest:main because it is valid. Remove invalid tool entry points in favor of existing Make targets; add worker/beat callables. Remove redundant asyncio and duplicate gepa, remove unused perfect, and place unused development/evaluation tools in appropriate dependency groups. Keep CPU torch pins because Docling depends on them transitively.

## Risks / Trade-offs

- [Cached resources cross fork] → PID-aware runtime reset and fresh-process/child tests; no construction during module import.
- [Slow startup kills Celery child] → lazy provisioning outside worker_process_init; provisioning failures stay observable at task execution.
- [Tests mock entire packages] → update fixtures and add subprocess checks so mocks cannot hide runtime import defects.
- [Internal import paths change] → migrate all source/test/operator references in one change; no duplicate compatibility task registrations.
- [Broad existing lint/type failures] → capture the baseline, fix migration regressions, report remaining unrelated failures precisely.
- [Multiple event loops remain for specialized capabilities] → each capability exclusively owns its resources and loop; generic task/Redis execution shares one runtime. One loop per resource lifetime is the invariant, not one global loop at any cost.

## Migration Plan

1. Capture baseline and complete/review specs before edits.
2. Separate contracts, producer configuration, and worker modules; relocate all feature tasks and their use cases.
3. Update imports, deployment entry points, tests, tooling, and operating documentation together.
4. Run registration, topology, retry, lifecycle, import-isolation, outbox, and full project checks.
5. Deploy API, workers and beat from the same revision during the next normal deployment. Existing queued messages remain valid. Roll back the code/image revision together; no schema rollback or queue renaming is needed.

## Source Verification

- Celery signals: https://docs.celeryq.dev/en/stable/userguide/signals.html#worker-process-init (slow handlers are not allowed; shutdown delivery is not guaranteed).
- Async resource loop affinity: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html#using-multiple-asyncio-event-loops.
- Existing source: database/base.py naming convention, app/config sections, app/shared/outbox, app/lifecycle/document_worker.py, app/connections/celery.py.
