"""Celery connection and configuration."""
import random

from celery import Celery

from app.config import get_settings


def create_celery_app() -> Celery:
    """Create and configure Celery application."""
    settings = get_settings()

    app = Celery(
        main="langchain_fastapi",
        broker=settings.RABBITMQ_URL,
        backend="rpc://",
        include=["app.tasks"],
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        broker_connection_retry_on_startup=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_default_retry_delay=5,
        # task_annotations={"*": {"max_retries": 5}},
        # retry_delay=random.uniform(1, 5),  # noqa: S311
    )

    return app


celery_app = create_celery_app()
