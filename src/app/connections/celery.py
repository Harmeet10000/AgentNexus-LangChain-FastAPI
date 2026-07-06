"""Celery connection and production reliability configuration."""

import time
from typing import Any, ClassVar, cast

import opentelemetry.trace as otel_trace
from celery import Celery, Task
from celery.signals import (
    after_task_publish,
    task_failure,
    task_postrun,
    task_prerun,
    task_retry,
    worker_shutting_down,
)
from kombu import Exchange, Queue
from opentelemetry import metrics
from redis.asyncio import Redis

from app.config import get_settings
from app.connections import create_redis_client
from app.shared.otel import setup_otel, shutdown_otel
from app.utils import logger

from .celery_reliability import (
    RedisClientProtocol,
    acquire_idempotency_lock,
    mark_idempotency_completed,
    mark_idempotency_failed_permanently,
    release_idempotency_processing_lock,
    run_with_circuit_breaker,
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

    def on_retry(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        _ = (args, kwargs, einfo)
        celery_task_retries_total.add(1, {"task_name": str(self.name)})
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).warning(f"Task scheduled for retry: {exc!s}")

    def on_failure(
        self,
        exc: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
        einfo: Any,
    ) -> None:
        _ = (args, kwargs, einfo)
        attrs = {"task_name": str(self.name), "status": "failure"}
        celery_task_completed_total.add(1, attrs)
        celery_task_duration.record(time.time() - self.request.started or time.time(), attrs)
        celery_task_retries_total.add(1, {"task_name": str(self.name)})
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).error(f"Task failed: {exc!s}")

    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: Any,
        kwargs: Any,
    ) -> None:
        _ = (retval, args, kwargs)
        attrs = {"task_name": str(self.name), "status": "success"}
        celery_task_completed_total.add(1, attrs)
        celery_task_duration.record(time.time() - self.request.started or time.time(), attrs)
        logger.bind(
            task=self.name,
            task_id=task_id,
            retry_count=self.request.retries,
        ).info("Task completed successfully")


def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    app = Celery(
        main="langchain_fastapi",
        broker=settings.RABBITMQ_URL,
        backend="rpc://",
        include=["tasks.auth_email_tasks", "tasks.example", "tasks.search_tasks"],
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
                name=settings.CELERY_DEAD_LETTER_QUEUE,
                exchange=TASK_DLX_EXCHANGE,
                routing_key=settings.CELERY_DEAD_LETTER_ROUTING_KEY,
                durable=True,
                queue_arguments={"x-queue-type": "quorum"},
            ),
        ),
        task_routes={
            "tasks.*": {
                "queue": settings.CELERY_DEFAULT_QUEUE,
                "routing_key": settings.CELERY_DEFAULT_ROUTING_KEY,
            }
        },
    )

    return app


celery_app = create_celery_app()

# OTel setup for Celery worker
if settings.OTEL_ENABLED:
    setup_otel(service_name="langchain-fastapi-celery")

_otel_celery_meter = metrics.get_meter("celery")
celery_task_completed_total = _otel_celery_meter.create_counter(
    "celery.task.completed_total", unit="1"
)
celery_task_duration = _otel_celery_meter.create_histogram("celery.task.duration_seconds", unit="s")
celery_task_retries_total = _otel_celery_meter.create_counter("celery.task.retries_total", unit="1")


@worker_shutting_down.connect
def _celery_shutdown_otel(**_: Any) -> None:
    shutdown_otel()


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
