"""RabbitMQ queue topology shared by producers and workers."""

from kombu import Exchange

from app.config import get_settings
from app.shared.jobs.names import CRAWLER_CRAWL, INGESTION_TASK_NAMES
from app.shared.jobs.registry import CeleryTaskRegistry

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
    default_route = {
        "queue": settings.CELERY_DEFAULT_QUEUE,
        "routing_key": settings.CELERY_DEFAULT_ROUTING_KEY,
    }
    ingestion_route = {
        "queue": settings.CELERY_INGESTION_QUEUE,
        "routing_key": settings.CELERY_INGESTION_ROUTING_KEY,
    }
    crawler_route = {
        "queue": settings.CELERY_CRAWLER_QUEUE,
        "routing_key": settings.CELERY_CRAWLER_ROUTING_KEY,
    }
    return {
        name: dict(
            ingestion_route
            if name in INGESTION_TASK_NAMES
            else crawler_route
            if name == CRAWLER_CRAWL
            else default_route
        )
        for name in CeleryTaskRegistry.registered_names()
    }
