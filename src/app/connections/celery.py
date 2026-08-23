"""Celery connection and production reliability configuration."""

import time
from typing import Any, ClassVar, cast, override

import opentelemetry.trace as otel_trace
from celery import Celery, Task
from celery.schedules import crontab
from celery.signals import (
    after_task_publish,
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
)
from kombu import Exchange, Queue
from opentelemetry import metrics
from redis.asyncio import Redis

from app.config import get_settings
from app.connections.redis import create_redis_client
from app.utils import logger

from .celery_reliability import (
    RedisClientProtocol,
    acquire_idempotency_lock,
    mark_idempotency_completed,
    mark_idempotency_failed_permanently,
    release_idempotency_processing_lock,
    run_with_circuit_breaker,
)
from .celery_task_names import (
    BILLING_DUNNING,
    BILLING_INVOICE_GENERATION,
    BILLING_RECEIPT_GENERATION,
    BILLING_RECONCILIATION,
    CREDITS_EXPIRE,
    CREDITS_RECONCILE,
    INGESTION_TASK_NAMES,
    TASK_DECLARING_MODULES,
)

settings = get_settings()

TASK_EXCHANGE = Exchange(
    name=settings.CELERY_DEFAULT_EXCHANGE,
    type="direct",
    durable=True,
)
TASK_DLX_EXCHANGE = Exchange(
    name=settings.CELERY_DEAD_LETTER_EXCHANGE,
    type="direct",
    durable=True,
)


def _task_routes() -> dict[str, dict[str, str]]:
    """Give every dispatchable name an explicit destination — no glob, no fallthrough.

    What this replaced was a single ``tasks.*`` glob, and the glob was close to
    decorative: only 5 of the 16 declared names begin with ``tasks.``, so the
    other 11 matched no route at all and arrived on the default queue through
    ``task_default_queue`` instead. Two mechanisms delivering to one queue read as
    one mechanism until the day a name needs a different queue, and then the
    question "which of these decides?" has to be answered from Celery's internals
    rather than from this file.

    Now the table is built from the single task-name definition site, so a name
    added there cannot be dispatched to an unrouted destination, and the answer to
    "where does this task go" is visible without knowing anything about matching
    order. Measured while making the change, and recorded because the tempting
    smaller edit — three exact entries left sitting beside the glob that also
    matches them — depends on it: ``MapRoute`` splits the mapping in its
    constructor, exact keys into one dict and globs into another, and consults the
    exact dict first, so exact names do win regardless of the order they are
    written in. That is a fact about a library version, not about this
    configuration, so the configuration no longer relies on it.

    Each name gets its own copy of the route mapping. ``Router.expand_destination``
    **pops** ``queue`` out of the dict it is handed; today ``MapRoute`` hands it a
    fresh copy so shared dicts would survive, but one shared dict emptied by the
    first lookup is a failure that would present as "routing works once".
    """
    default_route = {
        "queue": settings.CELERY_DEFAULT_QUEUE,
        "routing_key": settings.CELERY_DEFAULT_ROUTING_KEY,
    }
    ingestion_route = {
        "queue": settings.CELERY_INGESTION_QUEUE,
        "routing_key": settings.CELERY_INGESTION_ROUTING_KEY,
    }
    return {
        name: dict(ingestion_route if name in INGESTION_TASK_NAMES else default_route)
        for name in TASK_DECLARING_MODULES
    }


class ResilientTask(Task):
    """Base Celery task with retries, observability, and reliability helpers."""

    abstract = True
    autoretry_for: ClassVar[tuple[type[BaseException], ...]] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    retry_backoff: ClassVar[bool] = True
    retry_backoff_max: ClassVar[int] = settings.CELERY_RETRY_BACKOFF_MAX
    retry_jitter: ClassVar[bool] = True
    retry_kwargs: ClassVar[dict[str, int]] = {"max_retries": settings.CELERY_RETRY_MAX_RETRIES}

    _redis_client: ClassVar[Redis | None] = None

    @classmethod
    def get_redis_client(cls) -> RedisClientProtocol:
        if cls._redis_client is None:
            cls._redis_client = create_redis_client(settings.REDIS_URL)
        return cast("RedisClientProtocol", cls._redis_client)

    def acquire_idempotency_lock(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return acquire_idempotency_lock(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def mark_idempotency_completed(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        mark_idempotency_completed(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def mark_idempotency_failed_permanently(
        self,
        idempotency_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        mark_idempotency_failed_permanently(
            self.get_redis_client(),
            idempotency_key,
            task_id=self.request.id,
            ttl_seconds=settings.CELERY_IDEMPOTENCY_TTL_SECONDS,
            metadata=metadata,
        )

    def release_idempotency_processing_lock(self, idempotency_key: str) -> None:
        release_idempotency_processing_lock(self.get_redis_client(), idempotency_key)

    def run_with_circuit_breaker(
        self,
        name: str,
        operation,
    ) -> Any:
        return run_with_circuit_breaker(
            self.get_redis_client(),
            name,
            operation,
            failure_threshold=settings.CELERY_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout_seconds=settings.CELERY_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        )

    @override
    def on_retry(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        _ = (args, kwargs, einfo)
        _celery_meters()[2].add(1, {"task_name": str(self.name)})
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).warning(f"Task scheduled for retry: {exc!s}")

    @override
    def on_failure(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        _ = (args, kwargs, einfo)
        completed, duration, retries = _celery_meters()
        attrs = {"task_name": str(self.name), "status": "failure"}
        completed.add(1, attrs)
        duration.record(time.time() - self.request.started or time.time(), attrs)
        retries.add(1, {"task_name": str(self.name)})
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).error(f"Task failed: {exc!s}")

    @override
    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
    ) -> None:
        _ = (retval, args, kwargs)
        completed, duration, _ = _celery_meters()
        attrs = {"task_name": str(self.name), "status": "success"}
        completed.add(1, attrs)
        duration.record(time.time() - self.request.started or time.time(), attrs)
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).info("Task completed successfully")


def create_celery_app() -> Celery:
    """Create and configure Celery application.

    Every module declaring a task is named in ``include`` — the list below is the
    authoritative one an operator reads, and a unit test asserts it agrees with
    the task-name definition module rather than either being derived from the
    other. Four modules were missing before, among them the unified document
    ingestion task, and they were registered only because the task package's
    initialiser happens to import them: importing any listed sibling imports that
    initialiser first, which imported the rest. Registration therefore rested on
    an import side effect in a file with no reason to know it was load-bearing,
    and tidying that file would have silently stopped ingestion from being
    consumed while dispatch carried on succeeding.

    ``include`` is lazy — Celery imports these when the application is finalised
    (worker start), not when it is constructed — so listing a module here costs a
    dispatching process nothing. The consequence is that a process which only
    dispatches holds no payload registrations at all; the typed dispatch helper
    handles that itself rather than forcing the whole list to be imported.

    The typed reference implementation of the two email tasks is deliberately
    absent: it declares the same two task names as the live email module, so
    listing both would make the winner depend on import order and silently
    replace a live implementation with a demonstration of one.
    """
    app = Celery(
        main="langchain_fastapi",
        broker=settings.RABBITMQ_URL,
        backend="rpc://",
        include=[
            "tasks.auth_email_tasks",
            "tasks.billing_tasks",
            "tasks.credit_tasks",
            "tasks.document_extraction_tasks",
            "tasks.document_tasks",
            "tasks.example",
            "tasks.pageindex_tasks",
            "tasks.search_tasks",
        ],
    )

    app.Task = ResilientTask
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=None,
        broker_transport_options={"confirm_publish": True},
        task_publish_retry=True,
        task_publish_retry_policy={
            "max_retries": 3,
            "interval_start": 0.25,
            "interval_step": 0.5,
            "interval_max": 5,
        },
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=settings.CELERY_WORKER_MAX_TASKS_PER_CHILD,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_default_retry_delay=settings.CELERY_DEFAULT_RETRY_DELAY,
        task_track_started=True,
        task_send_sent_event=True,
        worker_send_task_events=True,
        task_default_delivery_mode="persistent",
        task_default_queue=settings.CELERY_DEFAULT_QUEUE,
        task_default_exchange=settings.CELERY_DEFAULT_EXCHANGE,
        task_default_exchange_type="direct",
        task_default_routing_key=settings.CELERY_DEFAULT_ROUTING_KEY,
        task_create_missing_queues=False,
        task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
        task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
        result_expires=settings.CELERY_TASK_RESULT_EXPIRES,
        # Three queues, and a worker must name the one it wants with `-Q`. Without
        # `-Q` Celery consumes **every** queue declared here — measured, not
        # assumed — which means a bare worker also drains the dead-letter queue and
        # re-runs the very messages that were parked for a human to look at. The
        # deployed services and the documented commands all carry `-Q` for that
        # reason; it is not there to save a connection.
        #
        # The ingestion queue dead-letters to the same exchange and routing key as
        # the default one, so ingestion failures park in the existing dead-letter
        # queue rather than a fourth queue nobody watches. A separate ingestion
        # dead-letter queue was the alternative and was rejected: a dead-letter
        # queue with no consumer and no dashboard is indistinguishable from a
        # message that vanished, and the task name inside each parked message is
        # already enough to tell ingestion failures from billing ones.
        task_queues=(
            Queue(
                name=settings.CELERY_DEFAULT_QUEUE,
                exchange=TASK_EXCHANGE,
                routing_key=settings.CELERY_DEFAULT_ROUTING_KEY,
                durable=True,
                queue_arguments={
                    "x-queue-type": "quorum",
                    "x-dead-letter-exchange": settings.CELERY_DEAD_LETTER_EXCHANGE,
                    "x-dead-letter-routing-key": settings.CELERY_DEAD_LETTER_ROUTING_KEY,
                },
            ),
            Queue(
                name=settings.CELERY_INGESTION_QUEUE,
                exchange=TASK_EXCHANGE,
                routing_key=settings.CELERY_INGESTION_ROUTING_KEY,
                durable=True,
                queue_arguments={
                    "x-queue-type": "quorum",
                    "x-dead-letter-exchange": settings.CELERY_DEAD_LETTER_EXCHANGE,
                    "x-dead-letter-routing-key": settings.CELERY_DEAD_LETTER_ROUTING_KEY,
                },
            ),
            Queue(
                name=settings.CELERY_DEAD_LETTER_QUEUE,
                exchange=TASK_DLX_EXCHANGE,
                routing_key=settings.CELERY_DEAD_LETTER_ROUTING_KEY,
                durable=True,
                queue_arguments={"x-queue-type": "quorum"},
            ),
        ),
        task_routes=_task_routes(),
        beat_schedule={
            "billing-invoice-daily": {
                "task": BILLING_INVOICE_GENERATION,
                "schedule": crontab(hour=0, minute=15),
            },
            "billing-dunning-daily": {
                "task": BILLING_DUNNING,
                "schedule": crontab(hour=1, minute=0),
            },
            "billing-receipt-daily": {
                "task": BILLING_RECEIPT_GENERATION,
                "schedule": crontab(hour=1, minute=45),
            },
            "billing-reconciliation-daily": {
                "task": BILLING_RECONCILIATION,
                "schedule": crontab(hour=2, minute=0),
            },
            "credits-expire-daily": {
                "task": CREDITS_EXPIRE,
                "schedule": crontab(
                    hour=settings.CREDIT_EXPIRATION_CRON_HOUR,
                    minute=settings.CREDIT_EXPIRATION_CRON_MINUTE,
                ),
            },
            "credits-reconcile-weekly": {
                "task": CREDITS_RECONCILE,
                "schedule": crontab(
                    hour=settings.CREDIT_RECONCILIATION_CRON_HOUR,
                    minute=settings.CREDIT_RECONCILIATION_CRON_MINUTE,
                    day_of_week=settings.CREDIT_RECONCILIATION_CRON_DAY_OF_WEEK,
                ),
            },
        },
    )

    return app


celery_app = create_celery_app()

# OTel meters are created lazily on first task event; importing this module
# must not spin up exporters or the app.shared.otel package.
_otel_celery_meters: tuple[Any, Any, Any] | None = None


def _celery_meters() -> tuple[Any, Any, Any]:
    global _otel_celery_meters  # noqa: PLW0603 — module-level lazy init
    if _otel_celery_meters is None:
        meter = metrics.get_meter("celery")
        _otel_celery_meters = (
            meter.create_counter("celery.task.completed_total", unit="1"),
            meter.create_histogram("celery.task.duration_seconds", unit="s"),
            meter.create_counter("celery.task.retries_total", unit="1"),
        )
    return _otel_celery_meters


@after_task_publish.connect
def log_task_published(
    sender: str | None = None,
    headers: dict[str, Any] | None = None,
    exchange: str | None = None,
    routing_key: str | None = None,
    **_: Any,
) -> None:
    logger.bind(
        task=sender,
        task_id=(headers or {}).get("id"),
        exchange=exchange,
        routing_key=routing_key,
    ).info("Celery task published")


@task_prerun.connect
def log_task_prerun(
    task_id: str | None = None,
    task: Task | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {
        "task": task.name if task else None,
        "task_id": task_id,
        "args_count": len(args or ()),
        "kwargs_keys": sorted((kwargs or {}).keys()),
    }
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).info("Celery task started")


@task_postrun.connect
def log_task_postrun(
    task_id: str | None = None,
    task: Task | None = None,
    state: str | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {"task": task.name if task else None, "task_id": task_id, "state": state}
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).info("Celery task finished")


@task_retry.connect
def log_task_retry(
    request: Any | None = None,
    reason: BaseException | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {
        "task": getattr(request, "task", None),
        "task_id": getattr(request, "id", None),
        "retry_count": getattr(request, "retries", None),
    }
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).warning(f"Celery task retry emitted: {reason!s}")


@task_failure.connect
def log_task_failure(
    task_id: str | None = None,
    exception: BaseException | None = None,
    sender: Task | None = None,
    **_: Any,
) -> None:
    span_ctx = otel_trace.get_current_span().get_span_context()
    trace_id = format(span_ctx.trace_id, "032x") if span_ctx.is_valid else None
    extra = {"task": sender.name if sender else None, "task_id": task_id}
    if trace_id:
        extra["trace_id"] = trace_id
    logger.bind(**extra).error(f"Celery task failed signal: {exception!s}")
