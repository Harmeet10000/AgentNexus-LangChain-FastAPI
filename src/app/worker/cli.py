"""Installed worker and scheduler entry points."""

import sys

from app.config import get_settings
from app.worker.celery_app import celery_app


def main() -> None:
    """Start a prefork worker with explicit queue selection supplied by the caller."""
    celery_app.worker_main(["worker", "-Q", get_settings().CELERY_DEFAULT_QUEUE, *sys.argv[1:]])


def beat() -> None:
    """Start the periodic scheduler."""
    celery_app.start(["beat", *sys.argv[1:]])
