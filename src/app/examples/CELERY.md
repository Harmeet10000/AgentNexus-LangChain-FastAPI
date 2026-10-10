# Celery runtime and feature tasks

The API publishes through `app.connections.celery_producer:celery_app`. Workers and beat load
`app.worker.app:celery_app`. Both use the same RabbitMQ topology, but only the
worker application includes task implementations, schedules and execution signals.
Importing the producer or validating a payload must never load `app.worker` or a
feature's `tasks.py`.

## Ownership

| Location | Responsibility |
|---|---|
| `app.shared.jobs.names` | Stable wire names; changing a Python path never changes these |
| `app.shared.jobs.payloads` | Pure Pydantic message schemas |
| `app.shared.jobs.registry` | Static payload registry and dispatch validation |
| `app.shared.jobs.topology` | Shared RabbitMQ exchanges and explicit queue routes |
| `app.connections.celery_producer` | Producer application and transport factory |
| `app.worker.app` | Worker application and explicit runtime configuration |
| `app.worker.task_modules` | All feature consumer modules and name-to-module mapping |
| `app.worker.schedules` | Beat schedule with arguments matching consumer signatures |
| `app.worker.runtime` | Persistent loop for sequential prefork/solo async execution |
| `app.worker.documents`, `app.worker.crawler` | Capability resources and owning loops |
| `app.features.<owner>.tasks` | Celery entry points, invoking feature services/jobs |
| `app.features.billing.jobs`, `app.features.billing.credits.jobs` | Scheduled domain operations |

## Run and inspect

```bash
make celery              # default queue
make celery-ingestion    # document/model processing
make celery-crawler      # browser work
make celery-beat         # scheduler; publishes, never consumes
make celery-command      # print deployment commands
uv run worker --loglevel=info -Q default --concurrency=8
uv run beat --loglevel=info
uv run replay-outbox --help
uv run celery -A app.worker.app:celery_app inspect registered
uv run celery -A app.worker.app:celery_app inspect active_queues
```

RabbitMQ remains the broker, with quorum queues and publisher confirms. Redis serves
application caching and reliability operations, not broker storage. Queue names remain
`default`, `ingestion`, `crawler`, and the existing dead-letter queue. Existing messages
remain compatible with relocated task code. Raw `celery worker` commands must supply
`-Q`: otherwise Celery may also consume declared dead-letter queues. The installed
`worker` command defaults to `default`; explicit `-Q` selects another queue.

Use prefork (the deployment default) or solo. Thread, eventlet and gevent pools are not
supported by the synchronous persistent-loop bridge. Maintain separate consumers for
long ingestion, browser work and short jobs; prefetch=1 alone does not isolate running
work. Deploy API, workers and beat from the same revision; roll back their image revision
together. No queue or database migration is needed.

## Add a task

1. Define its wire name in `app.shared.jobs.names`.
2. Put the payload schema in `app.shared.jobs.payloads` and add its model to the static
   registry. Models have no imports from features, connections or worker code.
3. Add a thin entry point in the owning feature's `tasks.py`. Import `celery_app` from
   `app.worker.app`, and `ResilientTask` from `app.worker.task`.
4. Add the task name and module to `TASK_DECLARING_MODULES`, and the module to
   `TASK_MODULES`. Add an explicit route and any beat entry with keyword arguments.
5. Run `make check-imports` and `uv run pytest tests/unit/celery`.

The payload registry is available at import time; producers never lazily import a
consumer to discover its schema. Use `CeleryTaskRegistry.typed_send(name, kwargs)` for
validated direct dispatch. Business writes that require durable delivery use the
existing transactional outbox in the same database transaction, preserving the stable
task name in the event. Never enqueue inside an uncommitted business transaction.

## Async resource lifetime

```python
from app.worker.runtime import run

# Inside a synchronous task boundary:
result = run(service_operation())
```

The generic runner lives for one child process and is reused by cached reliability
Redis operations and generic async jobs. Document and crawler capabilities each own
an explicit runner and their resource graph. No resource crosses between these loops.
Child initialization handlers only reset inherited state. First use provisions slow
resources outside the short Celery initialization-handler deadline. Normal shutdown
closes clients before their owning loops and tolerates repeated cleanup. SIGKILL does
not guarantee shutdown callbacks, so correctness cannot depend on a final callback.

Do not use `asyncio.run()` with a cached async client: each call creates another loop.
Per-operation engines/clients must still close in `finally` on the loop that created
them. Keep transaction ownership in the feature use case, with rollback on failure.

## Errors, retries and duplicate delivery

Services return their feature's typed Result union. Task adapters unwrap it with
`isinstance(result, Failure)`; retryable failures raise the appropriate infrastructure
exception or Celery retry signal. Permanent failures are recorded explicitly. Never
raise a FeatureError model or return an HTTP response from a task.

`ResilientTask` retains configured exponential backoff, jitter, retry limits and
late acknowledgement. Observability initializes in each child, and publication trace
headers are injected before the broker send. Log context fields rather than payloads
containing secrets.

Delivery is at least once. Use a stable business operation key, durable uniqueness or
operation state, and provider idempotency when an external side effect requires it.
The outbox closes the commit-versus-publish gap; it can still republish after a crash.
A Redis lease is only a concurrency aid: it can expire, be evicted, or outlive a crashed
execution. It is not proof an external charge or email occurred exactly once. Preserve
the document pipeline's durable advisory-lock/idempotency checks. RPC task results are
short-lived observations, not a business audit trail.
