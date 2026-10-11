"""Producer-only Celery application and shared transport configuration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from celery import Celery
from celery.exceptions import CeleryError
from celery.signals import before_task_publish
from kombu import Queue
from kombu.exceptions import OperationalError

from app.config import get_settings
from app.shared.jobs.registry import CeleryTaskRegistry
from app.shared.jobs.telemetry import log_task_published
from app.shared.jobs.topology import TASK_DLX_EXCHANGE, TASK_EXCHANGE, task_routes
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from app.shared.jobs.payloads import CeleryTaskPayload

before_task_publish.connect(log_task_published)
settings = get_settings()


def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    app = Celery(
        main="langchain_fastapi",
        broker=settings.RABBITMQ_URL,
        backend="rpc://",
    )
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
                name=settings.CELERY_CRAWLER_QUEUE,
                exchange=TASK_EXCHANGE,
                routing_key=settings.CELERY_CRAWLER_ROUTING_KEY,
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
        task_routes=task_routes(),
    )
    return app


celery_app = create_celery_app()


async def dispatch(
    celery: Celery,
    task_name: str,
    payload: CeleryTaskPayload | dict[str, object],
    *,
    expires: float | None = None,
    countdown: float | None = None,
) -> str:
    """Validate a payload against the registry and publish off the event loop.

    This is the single dispatch path: producers never publish directly, and
    never pass ``queue=`` — routing lives in ``task_routes()`` alone.
    Returns the broker task id.
    """
    kwargs = payload.model_dump(mode="json") if not isinstance(payload, dict) else payload
    CeleryTaskRegistry.validate(task_name, kwargs)
    try:
        result = await asyncio.to_thread(
            celery.send_task,
            task_name,
            kwargs=kwargs,
            expires=expires,
            countdown=countdown,
        )
    except (CeleryError, ConnectionError, OSError, TimeoutError, OperationalError) as exc:
        exc.add_note(f"task={task_name}, operation=celery_dispatch")
        message = "Could not enqueue background work"
        raise ServiceUnavailableException(message) from exc
    return result.id
