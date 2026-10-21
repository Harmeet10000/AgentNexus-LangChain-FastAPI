"""RabbitMQ queue topology shared by producers and workers."""

from kombu import Exchange

from app.config import get_settings
from app.shared.jobs.catalog import JOB_CATALOG, JobQueue

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
