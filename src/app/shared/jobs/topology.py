"""RabbitMQ queue topology shared by producers and workers."""

from typing import TYPE_CHECKING

from kombu import Exchange, Queue

from app.config import get_settings
from app.shared.jobs.catalog import JOB_CATALOG, JobQueue

if TYPE_CHECKING:
    from kombu import Connection

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

#: The terminal DLQ (G6, 2026-11-15 sweep): classic (not quorum) so it
#: supports message TTL on RabbitMQ's managed version, capped by
#: ``x-max-length`` with ``reject-publish`` and a 7-day message TTL so
#: forensics parking can never grow unbounded.  Its binding routing key must
#: stay equal to the ``x-dead-letter-routing-key`` every work queue carries
#: (see ``dead_letter_queue`` consumers below) — that equality IS the
#: dead-lettering route; a drift drops poisons on the broker floor again.
DLQ_MAX_LENGTH = 10_000
DLQ_MESSAGE_TTL_MS = 7 * 24 * 60 * 60 * 1_000


def dead_letter_queue() -> Queue:
    """Build the DLQ exactly as both the declaration and ``task_queues`` see it.

    One constructor shared by the producer's ``task_queues`` entry and the
    idempotent startup declaration, so the two cannot drift.
    """
    return Queue(
        name=settings.CELERY_DEAD_LETTER_QUEUE,
        exchange=TASK_DLX_EXCHANGE,
        routing_key=settings.CELERY_DEAD_LETTER_ROUTING_KEY,
        durable=True,
        queue_arguments={
            "x-queue-type": "classic",
            "x-max-length": DLQ_MAX_LENGTH,
            "x-overflow": "reject-publish",
            "x-message-ttl": DLQ_MESSAGE_TTL_MS,
        },
    )


def declare_dead_letter_topology(connection: "Connection") -> None:
    """Idempotently declare the DLX exchange, the DLQ queue, and the binding.

    G6: the exchange/queue existed only inside ``task_queues`` — no worker
    consumes the DLQ and producers never declare it, so RabbitMQ had no
    ``tasks.dlx`` at all and every ``Reject(requeue=False)`` poison was
    silently discarded.  This runs once per producer process before the first
    publish and once per worker boot; RabbitMQ declares are idempotent.
    """
    with connection.channel() as channel:
        TASK_DLX_EXCHANGE.declare(channel=channel)
        # Queue.declare also binds the queue to its exchange with its routing key.
        queue = dead_letter_queue()
        queue.maybe_bind(channel)
        queue.declare()


def task_routes() -> dict[str, dict[str, str]]:
    """Give every dispatchable name an explicit destination — no glob, no fallthrough."""
    routes = {
        JobQueue.DEFAULT: {
            "queue": settings.CELERY_DEFAULT_QUEUE,
            "routing_key": settings.CELERY_DEFAULT_ROUTING_KEY,
        },
        JobQueue.INGESTION: {
            "queue": settings.CELERY_INGESTION_QUEUE,
            "routing_key": settings.CELERY_INGESTION_ROUTING_KEY,
        },
        JobQueue.CRAWLER: {
            "queue": settings.CELERY_CRAWLER_QUEUE,
            "routing_key": settings.CELERY_CRAWLER_ROUTING_KEY,
        },
    }
    return {name: dict(routes[definition.queue]) for name, definition in JOB_CATALOG.items()}
