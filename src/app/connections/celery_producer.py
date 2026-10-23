"""Producer-only Celery application and shared transport configuration."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, cast, override

from celery import Celery
from celery.exceptions import CeleryError
from celery.signals import before_task_publish
from kombu import Queue
from kombu.exceptions import OperationalError

from app.config import get_settings
from app.shared.jobs import DirectPublicationAdapter as DirectPublicationPort
from app.shared.jobs.telemetry import log_task_published
from app.shared.jobs.topology import TASK_DLX_EXCHANGE, TASK_EXCHANGE, task_routes
from app.utils import ServiceUnavailableException

if TYPE_CHECKING:
    from typing import Any

    from app.shared.jobs import EnqueueOptions, TypedJob

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
        # Late acknowledgements are retained for worker loss, but ordinary
        # task exceptions/timeouts are acknowledged once and sent through the
        # task retry/dead-letter policy.  Requeueing every failure here creates
        # an unbounded redelivery loop for malformed or permanently failed jobs.
        task_acks_on_failure_or_timeout=True,
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


class _BrokerResult(Protocol):
    id: str


class DirectPublicationAdapter(DirectPublicationPort):
    """Publish catalog-validated jobs through one configured Celery app.

    The adapter owns the transport-specific details (JSON-compatible kwargs,
    broker call off the event loop, and the project's service-unavailable error
    mapping).  Queue selection is intentionally absent: Celery resolves it from
    the application route table built from the immutable job catalog. This is
    the transport side of the JobBus seam: producers reach publication through
    ``JobBus.enqueue``, never by constructing or calling this class directly.
    """

    def __init__(self, celery: Celery) -> None:
        self._celery = celery

    @override
    async def publish(
        self,
        job: TypedJob[Any],
        *,
        options: EnqueueOptions,
    ) -> str:
        """Validate then publish without blocking the caller's event loop."""
        validated = self._validate(job)
        transport_options = options.model_dump(exclude_none=True)
        result = await asyncio.to_thread(self._send_result_sync, validated, transport_options)
        return str(cast("_BrokerResult", result).id)

    def _send_result_sync(self, job: TypedJob[Any], transport_options: dict[str, object]) -> object:
        """Publish synchronously on a worker thread and return Celery's result.

        This is the raw transport call, deliberately private and the only place
        in the application allowed to call ``send_task``.  The adapter carries
        no queue or routing choice of its own: routing is resolved from the
        application route table built from the immutable job catalog.
        """
        try:
            return self._celery.send_task(
                job.task_name,
                kwargs=job.wire_payload(),
                **transport_options,
            )
        except (CeleryError, OperationalError, ConnectionError, OSError, TimeoutError) as exc:
            exc.add_note(f"task={job.task_name}, operation=celery_dispatch")
            message = "Could not enqueue background work"
            raise ServiceUnavailableException(message) from exc

    @staticmethod
    def _validate(job: TypedJob[Any]) -> TypedJob[Any]:
        """Cross the JobBus validation seam before any broker side effect."""
        from app.shared.jobs.bus import (  # noqa: PLC0415
            TypedJob,
            _validated_job,
        )

        # Pydantic's ``model_copy(update=...)`` intentionally does not
        # revalidate updates.  Rebuild such a forged/malformed job through the
        # catalog model first so invalid payloads still fail with the public
        # dispatch validation error, never with an AttributeError at send time.
        if not hasattr(job.payload, "model_dump"):
            job = TypedJob.from_definition(job.definition, job.payload)

        return _validated_job(job)


CeleryDirectPublicationAdapter = DirectPublicationAdapter
