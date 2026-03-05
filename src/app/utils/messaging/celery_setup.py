from celery import Celery

from app.config import get_settings
from app.utils import logger

settings = get_settings()

celery_app = Celery(
    main="worker",
    broker=settings.RABBITMQ_URL,
    backend=settings.RESULT_BACKEND,
    transport_options={"confirm_publish": True},
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # EDGE CASE: Ensure tasks aren't lost if a worker crashes
    task_acks_late=True,
    # EDGE CASE: Prevent memory leaks by restarting workers after X tasks
    worker_max_tasks_per_child=500,
    # PROD SETTING: Confirms task reached RabbitMQ
    broker_transport_options={"confirm_publish": True},
)

# 2. Configuration (Replaces your ExchangeTypes and durable setup)
celery_app.conf.task_queues = {
    "default": {
        "exchange": "tasks",
        "exchange_type": "direct",
        "binding_key": "tasks",
    },
}

# Optional: Global retry/backoff defaults
celery_app.conf.task_publish_retry_policy = {
    "max_retries": 3,
    "interval_start": 0.5,
    "interval_step": 0.5,
    "interval_max": 5,
}

@celery_app.task(
    bind=True,
    name="send_user_email",
    autoretry_for=(Exception,),  # Retry on any error
    retry_backoff=True,  # Exponential backoff (1s, 2s, 4s...)
    retry_backoff_max=600,  # Max wait 10 mins
    max_retries=5,  # Give up after 5 tries
)
def send_email_task(self, email: str, content: str) -> None:
    try:
        # Business logic here
        pass
    except Exception as exc:
        logger.error(f"Error sending email to {email}: {exc}")
        raise self.retry(exc=exc)
