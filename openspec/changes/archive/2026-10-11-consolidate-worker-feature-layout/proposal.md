> Change class: **L** — cross-cutting runtime and source-layout migration.

## Why

The modular monolith already owns its tables and services by feature, but one connection module mixes producer configuration, task execution, reliability state, registration, and telemetry. Separating producer contracts from worker execution makes API imports predictable and fixes pooled async Redis calls that currently create a new event loop on every operation.

## What Changes

- Adopt the supplied plan's feature ownership and dedicated worker runtime, adjusted to the repository's existing RabbitMQ, outbox, error unions, settings, and billing hierarchy.
- Move all six task modules into their owning features; separate scheduled billing/credit use cases from thin task entry points.
- Introduce shared, worker-independent task names, payload contracts, and typed dispatch; validation must not import task implementations.
- Give worker execution, reliability, observability, resource lifetimes, routing and schedules explicit homes; preserve wire names, queues, payloads, and retries.
- Correct packaging commands, first-party import classification, unused/backport dependencies and development dependency placement; enforce runtime boundaries in CI.
- **BREAKING** internal Python task module paths and the deployed Celery application reference change together. Durable message names and public HTTP APIs do not change.

## Scope / Non-Goals

Complete this migration in one change, including tests, commands, documentation, and verification. Preserve pre-existing working-tree changes. Do not switch RabbitMQ to Redis, rename queues/tables/environment variables, rewrite feature error handling, add speculative events, or pretend a Redis lease guarantees exactly-once effects. Keep the existing settings sections, SQLAlchemy naming convention, Alembic registration, shared result kernel, and outbox implementation. OCR packaging is not separable by moving direct torch declarations alone because Docling requires torch transitively.

## Capabilities

### New Capabilities

- `worker-runtime-isolation`: producer-safe payload validation and dispatch, feature-owned entry points, persistent worker event loops and mechanically checked import boundaries.

### Modified Capabilities

None. Existing celery-worker-deployment and transactional-outbox requirements remain applicable.

## Impact

Affected: app connections/shared jobs/worker/features, existing task tests, deployment commands, Makefile, pyproject/lockfile, CI and worker documentation. No database migration or external service writes are required. Add import-linter as a development tool.

## Risks

- Import cycles or missing registration: static payload registry, explicit consumer module list, fresh-process import and registration tests.
- Async resources crossing loops/forks: persistent child runtime, lazy capability initialization and deterministic teardown tests.
- Stranded messages: preserve every existing task wire name and queue; update all deployment references atomically.
- Existing workspace edits: retain their content and run baseline checks before implementation.
