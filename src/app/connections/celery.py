"""Celery connection and configuration."""

from celery import Celery

from app.config.settings import get_settings


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
    )

    return app


celery_app = create_celery_app()
