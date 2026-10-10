"""Worker application, explicitly loaded by CLI, Compose and beat."""

from app.connections.celery_producer import create_celery_app
from app.worker.schedules import BEAT_SCHEDULE
from app.worker.task import ResilientTask
from app.worker.task_modules import TASK_MODULES

celery_app = create_celery_app()
celery_app.Task = ResilientTask
celery_app.conf.update(
    include=TASK_MODULES, imports=("app.worker.signals",), beat_schedule=BEAT_SCHEDULE
)
